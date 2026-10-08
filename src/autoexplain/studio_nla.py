"""Session-local, explicitly prepared, CPU-staged Qwen L20 NLA workflow.

Nothing fetches artifacts, loads models or imports Streamlit at module import.
"""
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil

import torch

from .nla import (
    NLA_REPOSITORIES, NaturalLanguageAutoencoder, _checkpoint, _normalize,
    _weight_files, extract_nla_activations, nla_memory_preflight,
    nla_reconstruction_loss, patch_nla_activations, validate_nla_directory,
)
from .studio import fingerprint, safe_error


def _validate(path, role):
    if role in ("av", "ar"):
        root, _ = _checkpoint(path, role)
    else:
        root = validate_nla_directory(path, role)
        config = json.loads((root / "config.json").read_text())
        if (config.get("model_type") != "qwen2" or config.get("hidden_size") != 3584
                or config.get("num_hidden_layers") != 28
                or config.get("intermediate_size") != 18944
                or config.get("num_attention_heads") != 28
                or config.get("num_key_value_heads") != 4
                or config.get("vocab_size") != 152064
                or config.get("auto_map") or config.get("quantization_config")):
            raise ValueError("Only native unquantized Qwen2.5-7B-Instruct is supported")
        _weight_files(root, role)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    return str(root)


def _stamp(paths):
    """Path-free mutation fingerprint, not content authentication or a revision."""
    records = []
    for role, path in sorted(paths.items()):
        root = validate_nla_directory(path, role)
        for file in sorted(root.rglob("*")):
            if file.is_file():
                info = file.stat()
                records.append((role, str(file.relative_to(root)), info.st_size, info.st_mtime_ns))
    return fingerprint(records)


def _cached(role, revision):
    from huggingface_hub import snapshot_download
    return snapshot_download(NLA_REPOSITORIES[role], revision=revision, local_files_only=True)


# Upper bounds for remaining weights plus metadata/transfer headroom. Incomplete
# blobs receive no space credit: their sizes do not prove reusable contents.
_DISK_BUDGET_GIB = {"source": 36, "av": 36, "ar": 28}


class _DiskPreflightError(RuntimeError):
    """Static, path-free UI error for refused artifact transfers."""


def _cache_disk_free(role):
    """Inspect the actual future blob filesystem without creating directories."""
    from huggingface_hub.constants import HF_HUB_CACHE
    destination = (Path(HF_HUB_CACHE).expanduser() /
                   ("models--" + NLA_REPOSITORIES[role].replace("/", "--")) / "blobs")
    probe = destination.resolve()
    # A not-yet-created cache inherits the filesystem of its nearest existing
    # ancestor; an existing repository/blob mount can differ from home/cache.
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def prepare_artifact(role, revision="main", *, progress=None, cancel=None, override_disk=False):
    """Explicit owned foreground transfer, cancellable at 250 ms UI boundaries.

    Reuses validated complete assets. Interrupted Hub partial blobs are retained
    for Hub-managed resumption; no arbitrary URLs or repositories are accepted.
    Incomplete assets require a conservative role-specific free-disk budget on
    the actual blob filesystem, unless override_disk=True is explicitly chosen.
    """
    if role not in NLA_REPOSITORIES or not re.fullmatch(r"[A-Za-z0-9._/-]{1,100}", revision):
        raise ValueError("Unsupported artifact or revision")

    def boundary(message):
        if cancel is not None and cancel():
            raise InterruptedError("Preparation cancelled")
        if progress is not None:
            progress(message)

    boundary("Checking existing local assets")
    try:
        return _validate(_cached(role, revision), role)
    except (OSError, ValueError, KeyError):
        pass
    boundary("Checking actual Hub-cache filesystem free space")
    try:
        available = _cache_disk_free(role)
    except (OSError, RuntimeError):
        available = None
    required = _DISK_BUDGET_GIB[role] * 1024 ** 3
    if (available is None or available < required) and not override_disk:
        raise _DiskPreflightError(
            "Preparation stopped: cache-disk space is unknown or below the conservative "
            f"{_DISK_BUDGET_GIB[role]} GiB {role.upper()} budget. Free space or explicitly "
            "enable the disk-preflight override. No transfer was started.")
    import subprocess
    import sys
    import time
    # Child owns only a fixed-repository Hub operation. No detached process,
    # shell, remote code, pickle assets, or exception text is exposed to the UI.
    script = """
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=1,
    allow_patterns=['*.json', 'model.safetensors', 'model-*.safetensors',
                    'value_head.safetensors', '*.txt', '*.model', '*.jinja', 'nla_meta.yaml'])
"""
    boundary("Starting explicit Hub transfer")
    child = subprocess.Popen([sys.executable, "-c", script, NLA_REPOSITORIES[role], revision],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = time.monotonic()
    try:
        while child.poll() is None:
            boundary(f"Preparing {role.upper()} · {int(time.monotonic() - started)} s · Stop cancels transfer")
            time.sleep(0.25)
        if child.returncode:
            raise RuntimeError("Artifact preparation failed")
        boundary("Validating complete local assets")
        return _validate(_cached(role, revision), role)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def _source_stage(path, override, progress, action, cancel=None):
    """Scope the entire source lifetime, including cleanup on Streamlit Stop."""
    def boundary(message):
        if cancel is not None and cancel():
            raise InterruptedError("Source operation cancelled")
        progress(message)

    model = tokenizer = None
    try:
        boundary("Source: memory preflight (42 GiB additional RAM estimate)")
        nla_memory_preflight("source", override=override)
        root = _validate(path, "source")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        boundary("Source: loading CPU float32 weights; Stop takes effect after this loader call")
        local = dict(local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(root, **local)
        model, info = AutoModelForCausalLM.from_pretrained(
            root, torch_dtype=torch.float32, use_safetensors=True,
            output_loading_info=True, **local)
        if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Incomplete source weights")
        model.eval().requires_grad_(False)
        boundary("Source: ready")
        result = action(model, tokenizer, boundary)
        boundary("Source: operation complete; releasing weights")
        return result
    finally:
        model = tokenizer = None
        gc.collect()


def _scores(model, ids, target):
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite source output")
        target = int(logits.argmax()) if target is None else target
        return target, {"target_logit": float(logits[target]),
                        "target_probability": float(logits.softmax(-1)[target]),
                        "argmax_token_id": int(logits.argmax())}


def _summary(result):
    """Allowlisted numerical export: no prompts, explanations, paths or revisions."""
    return {key: result[key] for key in (
        "method", "positions", "target_id", "direction_mse", "cosine_similarity",
        "baseline", "comparison", "strength") if key in result}


def render():
    """Render the self-contained NLA page after the ordinary studio unloads."""
    import streamlit as st
    from . import studio_jobs

    st.header("Explore an activation in language")
    st.caption("Actual released Qwen2.5-7B L20 source → AV → AR → source intervention. Local CPU float32, one model at a time; no quantization.")
    st.warning("Large and slow: allow about 42 GiB AVAILABLE RAM for source/AV (34 GiB AR), plus complete checkpoint disk space. These are conservative estimates, not fit guarantees. A 10 GiB GPU is not the default execution target.")
    st.caption("Use Streamlit’s Stop to cancel. Transfers check every 250 ms; AV checks each generated token. A single model-loading call or source/AR forward cannot be interrupted here and may take minutes. All model references are released in finally blocks.")
    state = st.session_state.setdefault("nla_workspace", {"history": []})
    status = st.empty()

    def progress(message):
        studio_jobs.stage("nla", message)
        status.info(message)  # Main script thread; also a native Streamlit Stop boundary.

    if st.button("Unload / clear NLA session", key="nla_clear"):
        # Models are operation-scoped, never in cache_resource or session_state.
        for key in list(st.session_state):
            if key.startswith("nla_"):
                del st.session_state[key]
        gc.collect()
        st.rerun()

    st.subheader("1 · Prepare the three checkpoints")
    st.caption("No startup fetching. Use complete local directories or explicitly prepare the fixed repositories below. Configured Hub cache snapshots are reused without materializing duplicate weights. Local provenance is not authenticated.")
    paths, revisions = {}, {}
    with st.expander("Advanced · checkpoint paths and revisions", expanded=False):
        for role, repository in NLA_REPOSITORIES.items():
            st.write(f"**{role.upper()}** · `{repository}`")
            paths[role] = st.text_input(f"{role.upper()} local directory (blank = Hub cache)", key=f"nla_path_{role}")
            revisions[role] = st.text_input(f"{role.upper()} Hub revision", value="main", key=f"nla_revision_{role}")
    role = st.selectbox("Artifact to prepare from Hub", list(NLA_REPOSITORIES),
                        format_func=lambda value: f"{value.upper()} · {NLA_REPOSITORIES[value]}", key="nla_prepare_role")
    try:
        available_ram = nla_memory_preflight("source", override=True)["available_bytes"]
    except (OSError, RuntimeError):
        available_ram = None
    try:
        available_disk = _cache_disk_free(role)
    except (ImportError, OSError, RuntimeError):
        available_disk = None
    ram_label = f"{available_ram / 1024 ** 3:.1f} GiB" if available_ram is not None else "unknown"
    disk_label = f"{available_disk / 1024 ** 3:.1f} GiB" if available_disk is not None else "unknown"
    st.write(f"**Available RAM now:** {ram_label} · **Selected artifact’s cache-filesystem free space:** {disk_label}")
    st.caption("Live readings refresh on page rerun; no weights are loaded or downloaded to inspect resources. RAM estimates: source/AV 42 GiB, AR 34 GiB. Transfer budgets when incomplete: source/AV 36 GiB each, AR 28 GiB, including headroom; partial blobs receive no space credit. Complete validated snapshots are reused before any disk-budget refusal. These estimates do not reserve space or guarantee a fit.")
    override_disk = st.checkbox("Override low/unknown cache-disk preflight (I accept a failed transfer or full filesystem)",
                                key="nla_disk_override")
    consent_target = fingerprint([role, revisions[role]])
    if state.get("consent_target") != consent_target:
        st.session_state["nla_download_consent"] = False
        state["consent_target"] = consent_target
    consent = st.checkbox("I accept the selected model’s terms and explicitly authorize its large download", key="nla_download_consent")
    if st.button("Prepare selected artifact", disabled=not consent, key="nla_download"):
        try:
            prepare_artifact(role, revisions[role], progress=progress, override_disk=override_disk)
            status.success("Artifact prepared in the configured Hub cache. Validate all three below.")
        except _DiskPreflightError as exc:
            studio_jobs.failure("nla", str(exc))
            status.error(str(exc))  # Only our static, path-free preflight message.
        except Exception as exc:
            studio_jobs.failure("nla", safe_error(exc))
            status.error(safe_error(exc))
    st.caption("Prepare SOURCE, AV and AR in turn, or choose existing directories in Advanced; then validate all three to enable token selection. Preparation uses the Hub cache, not an Advanced local-directory override.")
    settings = fingerprint({"paths": paths, "revisions": revisions})
    if state.get("settings") != settings:
        state.clear()
        state.update(history=[], settings=settings, consent_target=consent_target)
    if st.button("Use / validate all three local checkpoints", key="nla_validate"):
        state.pop("paths", None)
        state.pop("tokens", None)
        state.pop("result", None)
        try:
            validated = {}
            for role in NLA_REPOSITORIES:
                progress("Validating local " + role.upper() + " assets (no weights loaded)")
                validated[role] = _validate(paths[role].strip() or _cached(role, revisions[role]), role)
            stamp = _stamp(validated)
            if state.get("stamp") != stamp:
                state["history"] = []
            state.update(paths=validated, stamp=stamp)
            status.success("All three complete local checkpoint sets validated; no model is resident.")
        except Exception as exc:
            studio_jobs.failure("nla", safe_error(exc))
            status.error(safe_error(exc))
    if "paths" not in state:
        st.info("Prepare or select the actual source, AV and AR checkpoints, then validate them to continue.")
        return

    st.success("Artifacts ready · models unloaded between stages")
    override = st.checkbox("Override conservative memory preflight (I accept process/OS out-of-memory risk)", key="nla_memory_override")
    prompt = st.text_area("Source text · native tokenizer, no chat template", value="The rabbit hopped into the garden.", max_chars=4000, key="nla_prompt")
    target_text = st.text_input("Fixed next-token target (optional; exactly one token). Blank locks the original source argmax.", max_chars=100, key="nla_target")
    budget = st.slider("AV generation budget per selected token", 32, 256, 200, key="nla_budget")
    input_key = fingerprint([prompt, target_text, budget, state["stamp"]])
    # Remove old dynamic widget drafts instead of accumulating source/explanation
    # text indefinitely across input changes in this browser session.
    for key in list(st.session_state):
        if key.startswith("nla_positions_") and key != "nla_positions_" + input_key[:16]:
            del st.session_state[key]
    if state.get("input_key") != input_key:
        state.pop("tokens", None)
        state.pop("result", None)
        state["input_key"] = input_key

    def unchanged():
        try:
            same = _stamp(state["paths"]) == state["stamp"]
        except (OSError, ValueError):
            same = False
        if not same:
            state.pop("tokens", None)
            state.pop("result", None)
            state.pop("paths", None)
            state["history"] = []
            raise ValueError("Checkpoint assets changed; validate again")

    if st.button("Tokenize source locally", key="nla_tokenize"):
        try:
            unchanged()
            if not prompt.strip():
                raise ValueError("Empty source text")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(state["paths"]["source"], local_files_only=True, trust_remote_code=False)
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            if not 1 <= len(ids) <= 128:
                raise ValueError("Source must contain 1–128 tokens")
            target_ids = tokenizer.encode(target_text, add_special_tokens=False) if target_text else None
            if target_ids is not None and len(target_ids) != 1:
                raise ValueError("Target must be exactly one token")
            state["tokens"] = {"ids": ids, "labels": tokenizer.convert_ids_to_tokens(ids),
                               "target": target_ids[0] if target_ids else None}
            state.pop("result", None)
        except Exception as exc:
            studio_jobs.failure("nla", safe_error(exc))
            status.error(safe_error(exc))
    if "tokens" not in state:
        st.caption("Tokenize to select exact source positions; max 128 source tokens, no silent truncation.")
        return
    tokens = state["tokens"]
    st.subheader("2 · Select source positions and verbalize")
    st.dataframe([{"position": i, "token_id": value, "token": tokens["labels"][i]}
                  for i, value in enumerate(tokens["ids"])], hide_index=True)
    positions = st.multiselect("Source positions (1–4; includes any special tokens)", range(len(tokens["ids"])),
                               default=[len(tokens["ids"]) - 1], key="nla_positions_" + input_key[:16])
    positions = tuple(sorted(positions))
    run_key = fingerprint([input_key, positions])
    if state.get("run_key") != run_key:
        state.pop("result", None)
        state["run_key"] = run_key
    if st.button("Extract → verbalize → reconstruct", disabled=not 1 <= len(positions) <= 4, key="nla_run"):
        state.pop("result", None)
        nla = None
        try:
            unchanged()
            ids = torch.tensor([tokens["ids"]], dtype=torch.long)

            def extract(model, tokenizer, boundary):
                boundary("Source: capturing block-20 selected rows")
                acts = extract_nla_activations(model, ids, positions=positions)
                boundary("Source: scoring fixed next-token baseline")
                target, baseline = _scores(model, ids, tokens["target"])
                return acts, target, baseline, tokenizer.decode([target])

            acts, target, baseline, target_label = _source_stage(state["paths"]["source"], override, progress, extract)
            nla = NaturalLanguageAutoencoder(state["paths"]["av"], state["paths"]["ar"], max_new_tokens=budget)
            nla.load_av(override_memory=override, progress=progress)
            texts = nla.encode(acts)
            nla.load_ar(override_memory=override, progress=progress)
            prediction = nla.decode([text.explanation for text in texts])
            result = {"acts": acts, "texts": texts, "prediction": prediction,
                      "method": "Qwen2.5-7B L20 NLA; CPU float32; residual-preserving edit",
                      "positions": list(positions), "target_id": target, "target_label": target_label,
                      "direction_mse": nla_reconstruction_loss(prediction, acts.vectors).tolist(),
                      "cosine_similarity": torch.nn.functional.cosine_similarity(prediction, acts.vectors).tolist(),
                      "baseline": baseline}
            progress("Round trip complete; releasing AR")
            nla.release()
            state["result"] = result
            state["history"] = (state["history"] + [_summary(result)])[-8:]
            status.success("Real AV explanations and AR directional reconstruction computed; all models released.")
        except Exception as exc:
            studio_jobs.failure("nla", safe_error(exc))
            status.error(safe_error(exc))
        finally:
            if nla is not None:
                nla.release()

    result = state.get("result")
    current_edit_prefix = None
    if result is not None:
        text_key = hashlib.sha256(json.dumps([text.explanation for text in result["texts"]]).encode()).hexdigest()[:16]
        current_edit_prefix = f"nla_edit_{run_key[:16]}_{text_key}_"
    for key in list(st.session_state):
        if key.startswith("nla_edit_") and (current_edit_prefix is None or not key.startswith(current_edit_prefix)):
            del st.session_state[key]
    if result is None:
        return
    st.write(f"**Locked target:** token {result['target_id']} · {result['target_label']!r} at the next position after the entire source sequence")
    st.dataframe([{"source_position": position, "direction_MSE": mse, "cosine": cosine}
                  for position, mse, cosine in zip(result["positions"], result["direction_mse"], result["cosine_similarity"])], hide_index=True)
    st.caption("MSE compares directions normalized to √3584; MSE = 2(1 − cosine). This is not FVE, norm fidelity, semantic truth or proof of causal importance.")
    st.subheader("3 · Edit language and measure the same-input source effect")
    edited = []
    # Explanation hash prevents old widget drafts from attaching to a new AV run.
    for index, text in enumerate(result["texts"]):
        st.write(f"Position {text.source_position} · original explanation")
        st.write(text.explanation)
        edited.append(st.text_area(f"Edited explanation for position {text.source_position}", value=text.explanation,
                                   max_chars=3000, key=current_edit_prefix + str(index)))
    strength = st.slider("Residual-preserving edit strength", 0.0, 2.0, 1.0, 0.05, key="nla_strength")
    comparison_key = fingerprint([edited, strength])
    if result.get("comparison_key") != comparison_key:
        result.pop("comparison", None)
        result.pop("strength", None)
    st.caption("x′ = x + strength × ‖x‖ × (unit(AR(edit)) − unit(AR(original))). Compare identical source IDs and the locked target; patching is one uncached forward, not generated-answer steering.")
    if st.button("Decode edit → reload source → compare controls", key="nla_compare"):
        nla = None
        result.pop("comparison", None)
        try:
            unchanged()
            if any(not text.strip() for text in edited):
                raise ValueError("Edited explanations cannot be empty")
            nla = NaturalLanguageAutoencoder(state["paths"]["av"], state["paths"]["ar"], max_new_tokens=budget)
            nla.load_ar(override_memory=override, progress=progress)
            after = nla.decode(edited)
            acts, before = result["acts"], result["prediction"]
            norms = acts.vectors.norm(dim=-1, keepdim=True)
            replacement = acts.vectors + strength * norms * (_normalize(after, 1.0) - _normalize(before, 1.0))
            reconstructed = norms * _normalize(before, 1.0)
            nla.release()  # AR must be gone before source weights begin loading.
            ids = torch.tensor([acts.token_ids], dtype=torch.long)

            def compare(model, tokenizer, boundary):
                rows = {}
                boundary("Source comparison: unpatched baseline")
                _, rows["baseline"] = _scores(model, ids, result["target_id"])
                for name, vector in (("unchanged_self_patch", acts.vectors),
                                     ("norm_matched_reconstruction", reconstructed),
                                     ("residual_preserving_edit", replacement)):
                    boundary("Source comparison: " + name)
                    with patch_nla_activations(model, acts, vector):
                        _, rows[name] = _scores(model, ids, result["target_id"])
                    rows[name]["delta_target_logit"] = rows[name]["target_logit"] - rows["baseline"]["target_logit"]
                return rows

            rows = _source_stage(state["paths"]["source"], override, progress, compare)
            result.update(comparison=rows, comparison_key=comparison_key, strength=strength)
            state["history"] = (state["history"] + [_summary(result)])[-8:]
            status.success("Same-input, fixed-target comparisons complete; source released.")
        except Exception as exc:
            studio_jobs.failure("nla", safe_error(exc))
            status.error(safe_error(exc))
        finally:
            if nla is not None:
                nla.release()
    if "comparison" in result:
        st.dataframe([{"condition": name, **values} for name, values in result["comparison"].items()], hide_index=True)
        st.warning("Edited language can be out of distribution and change multiple concepts. Output changes are local measurements, not faithful semantic explanations or general behavioral guarantees.")
    from .studio_storage import save_control
    persisted = {"input": {"source_text": prompt, "token_ids": tokens["ids"], "target_text": target_text},
                 "model": {"repositories": dict(NLA_REPOSITORIES), "checkpoint_stamp": state["stamp"],
                           "device": "cpu", "dtype": "float32"},
                 "settings": {"generation_budget": budget, "positions": list(positions), "strength": strength},
                 "result": _summary(result),
                 "original_explanations": [text.explanation for text in result["texts"]],
                 "edited_explanations": edited}
    save_control("nla", persisted, key="nla_local", default_name="Activation language experiment")
    with st.expander("Session history and numerical export"):
        st.caption("At most 8 numerical records in this browser session. Input/checkpoint changes invalidate the current result; checkpoint changes clear history. Exports deliberately exclude source text, explanation text, paths and requested revisions.")
        st.json(state["history"])
        st.download_button("Export numerical history", json.dumps(state["history"], indent=2),
                           file_name="nla-metrics.json", mime="application/json", key="nla_export")
