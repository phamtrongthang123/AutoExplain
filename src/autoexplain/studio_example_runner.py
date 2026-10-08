"""Bounded cache-only Gemma rerun transport; importing this module loads no ML stack."""
import json
import os
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import time

MODEL_ID = "google/gemma-4-E2B"
REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
PAIR_LABELS = ("France → Germany", "Italy → Spain", "Japan → China")
LAYERS = (0, 8, 16, 24, 34)
MAX_BYTES = 1024 * 1024


def interpreter():
    """Only a checkout-local dedicated environment or the current interpreter."""
    checkout = Path(__file__).absolute().parents[2]
    candidate = checkout / ".venv-llm" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def execute(*, run=False, pair=0, layer=8, inputs=None, progress=None):
    """Owned foreground child; Stop at a progress callback always reaps it.

    The abstract socket prevents concurrent example jobs across app sessions on
    Linux, without a mutable lock file. It does not evict unrelated GPU processes.
    """
    if type(pair) is not int or pair not in range(3) or layer not in LAYERS:
        raise ValueError("Choose a supported example and layer")
    if inputs is not None and (not isinstance(inputs, (list, tuple)) or len(inputs) != 4
                              or any(not isinstance(x, str) or not x.strip() or len(x) > 2000 for x in inputs)):
        raise ValueError("Provide donor, recipient, target and foil (1–2000 characters each)")
    payload = json.dumps(inputs).encode()
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This dedicated runner currently requires Linux")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", USE_TF="0", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    args = [interpreter(), "-u", str(Path(__file__).absolute()),
            "run" if run else "check", str(pair), str(layer)]
    budget = 180 if run else 30
    child = None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as lock:
        try:
            lock.bind("\0autoexplain-gemma-example-" + str(os.getuid()))
        except OSError:
            raise RuntimeError("Another example check/rerun is active; no job was started") from None
        try:
            child = subprocess.Popen(args, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            child.stdin.write(payload)
            child.stdin.close()
            started, total, pending, result = time.monotonic(), 0, b"", None
            message = "Checking interpreter, dependencies, pinned cache and free resources"
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while selector.get_map() or child.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > budget:
                        raise RuntimeError(f"Example exceeded its {budget}-second budget; stopped")
                    if progress is not None:
                        progress(int(elapsed), message)
                    for key, _ in selector.select(timeout=0.2):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        total += len(chunk)
                        if total > MAX_BYTES:
                            raise RuntimeError("Example output exceeded the 1 MiB limit; stopped")
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            event = json.loads(line)
                            if event["kind"] == "error":
                                raise RuntimeError(event["message"])
                            if event["kind"] == "progress":
                                message = event["message"]
                            elif event["kind"] == "result":
                                result = event["value"]
                            elif event["kind"] == "readiness":
                                message = "Cache/dependencies/resources: " + json.dumps(event["value"])
                                if not run:
                                    result = event["value"]
                child.wait(timeout=max(0.1, budget - (time.monotonic() - started)))
                if child.returncode or result is None or pending:
                    raise RuntimeError("Example failed or returned an incomplete report; no result retained")
                return result
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                else:
                    child.wait()
                if child.stdout is not None:
                    child.stdout.close()


def _emit(kind, **values):
    print(json.dumps({"kind": kind, **values}, allow_nan=False), flush=True)


def _check():
    # Metadata validation precedes torch/transformers imports and all weight loads.
    from importlib.metadata import version, PackageNotFoundError
    dependencies = {}
    for name in ("transformers", "torch", "torchvision", "torchaudio", "huggingface-hub", "psutil", "accelerate"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "missing"
    compatible = (dependencies["transformers"].split(".")[:2] == ["5", "18"]
                  and dependencies["torch"] == "2.10.0+cu126"
                  and dependencies["torchvision"] == "0.25.0+cu126"
                  and dependencies["torchaudio"] == "2.10.0+cu126"
                  and "missing" not in dependencies.values())
    state = {"interpreter": sys.executable, "dependencies": dependencies,
             "compatible": compatible, "cache": "not checked", "resources": "not checked", "ready": False}
    if not compatible:
        return state
    import psutil
    import torch
    from huggingface_hub import snapshot_download
    try:
        snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
        required = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
        complete = all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in required)
    except Exception:
        complete = False
    state["cache"] = "Pinned assets present (not weight-authenticated)" if complete else "Pinned snapshot missing/incomplete; no download attempted"
    host = psutil.virtual_memory().available / 1024**3
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.mem_get_info(0)[0] / 1024**3 if cuda else 0
    bf16 = cuda and torch.cuda.is_bf16_supported()
    state["resources"] = {"host_available_gib": host, "gpu_free_gib": gpu, "cuda": cuda, "bf16": bf16,
                          "required_host_gib": 18, "required_gpu_gib": 6}
    state["ready"] = bool(complete and bf16 and host >= 18 and gpu >= 6)
    return state


def _main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("check", "run"):
        raise ValueError("Unsupported invocation")
    pair, layer = int(sys.argv[2]), int(sys.argv[3])
    if pair not in range(3) or layer not in LAYERS:
        raise ValueError("Unsupported predefined study")
    raw = sys.stdin.buffer.read(32769)
    if len(raw) > 32768:
        raise ValueError("Input budget exceeded")
    inputs = json.loads(raw) if raw else None
    if inputs is not None and (not isinstance(inputs, list) or len(inputs) != 4
                              or any(not isinstance(x, str) or not x.strip() or len(x) > 2000 for x in inputs)):
        raise ValueError("Invalid input payload")
    state = _check()
    _emit("readiness", value=state)
    if sys.argv[1] == "check":
        return
    if not state["ready"]:
        _emit("error", message="Cache, dependency or resource checks failed. Check readiness; nothing downloaded or loaded.")
        return
    import resource
    import torch
    import transformers
    # Use APIs from this installed package/checkout, not cwd or a user-supplied path.
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    from autoexplain.pretrained import load_gemma4, GEMMA4_REVISION
    from autoexplain.pretrained_study import CAPITAL_PAIRS, run_pretrained_study
    if GEMMA4_REVISION != REVISION:
        raise ValueError("Pinned revision mismatch")
    inputs = tuple(inputs) if inputs is not None else CAPITAL_PAIRS[pair]
    # Check native alignment and targets before allocating model weights.
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    donor_ids, recipient_ids = [tokenizer.encode(text, add_special_tokens=True) for text in inputs[:2]]
    error = None
    if not 2 <= len(donor_ids) <= 32 or not 2 <= len(recipient_ids) <= 32:
        error = "Shorten each prompt to 2–32 native tokens; no truncation is applied."
    elif len(donor_ids) != len(recipient_ids):
        error = "Use donor and recipient prompts with equal native token counts."
    else:
        changed = [i for i, (a, b) in enumerate(zip(donor_ids, recipient_ids)) if a != b]
        if len(changed) != 1 or changed[0] == len(donor_ids) - 1:
            error = "Change exactly one token before the final token; keep the answer-readout position unchanged."
    targets = [tokenizer.encode(text, add_special_tokens=False) for text in inputs[2:]]
    if any(len(ids) != 1 for ids in targets) or targets[0] == targets[1]:
        error = "Target and foil must be distinct single tokens. Try a leading space before each word."
    if error:
        _emit("error", message=error)
        return
    del tokenizer
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    _emit("progress", message="Loading pinned local BF16 weights; CPU per-layer embeddings / cuda:0 decoder")
    bundle = load_gemma4(allow_download=False, decoder_device="cuda:0")
    _emit("progress", message="Computing one fixed pair, signed route gradients, donor/self/random patches")
    report = run_pretrained_study(bundle, pairs=(inputs,), layers=(layer,),
                                  random_seeds=(7, 11, 19), include_gradients=True)
    report['limitations'][0] = "One user-selected contrast, not a benchmark; inputs were fixed before this run."
    report.update(origin="live cache-only rerun", total_seconds=time.monotonic() - started,
                  python_version=sys.version.split()[0], torch_version=str(torch.__version__),
                  transformers_version=transformers.__version__, threads=2,
                  cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    _emit("result", value=report)


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        _emit("error", message="Backend failed: incompatible dependencies, invalid cached weights, resource exhaustion or study validation. No download or fallback was attempted.")
        sys.exit(1)
