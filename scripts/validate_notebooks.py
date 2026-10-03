#!/usr/bin/env python3
"""Execute a group of offline tutorials sequentially and retain outputs in place.

Run with the environment that contains torch, sklearn, matplotlib, nbclient,
nbformat and ipykernel. Each fresh kernel uses this interpreter explicitly.
Prints JSON measurements to stdout; no downloads, installs, or report files.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


NAMES = (
    "01_cnn_attribution", "02_unet_segmentation", "03_transformer_patching",
    "04_diffusion_steering", "05_tree_tabular",
)


INTEGRATIONS = (
    "06_attribution_backends", "07_sae_features", "08_jacobian_lens",
    "09_openai_sparse_circuits", "10_goodfire_parameter_decomposition",
    "11_diffusers_image_steering", "12_representation_and_tabular_controls",
)


def main():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per cell in seconds")
    parser.add_argument("--group", choices=["base", "integrations", "tracing"], default="base")
    args = parser.parse_args()
    names = {"base": NAMES, "integrations": INTEGRATIONS, "tracing": ("13_circuit_tracing",)}[args.group]
    root = Path(__file__).resolve().parents[1]
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    reports = []
    os.environ["USE_TF"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    for name in names:
        path = root / "notebooks" / f"{name}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        manager = KernelManager(kernel_name="python3")
        # Only the in-memory spec is changed; never modify a shared kernelspec.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, timeout=args.timeout, km=manager,
                                resources={"metadata": {"path": str(root)}})
        started = time.perf_counter()
        try:
            # An externally supplied manager is explicitly shut down by this context.
            with client.setup_kernel(cleanup_kc=True):
                client.execute(cleanup_kc=False)
        except Exception:
            print(f"FAILED: {path}; outputs not overwritten", file=sys.stderr)
            raise
        execution_wall = time.perf_counter() - started
        metrics = None
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.execution_count is None:
                raise RuntimeError(f"Unexecuted cell in {path}")
            for output in cell.outputs:
                if output.output_type == "error":
                    raise RuntimeError(f"Notebook error in {path}")
                if output.output_type == "stream":
                    for line in output.text.splitlines():
                        if line.startswith('{"tutorial_compute_wall_seconds"'):
                            metrics = json.loads(line)
        if metrics is None:
            raise RuntimeError(f"Missing kernel self-measurement in {path}")
        report = {"notebook": path.name, "python_version": sys.version.split()[0],
                  "execution_wall_seconds": round(execution_wall, 3), **metrics}
        notebook.metadata["autoexplain_execution"] = report
        nbformat.write(notebook, path)
        reports.append(report)
        print(json.dumps(report), flush=True)
    print(json.dumps({"completed": len(reports), "all_notebooks_passed": True}))


if __name__ == "__main__":
    main()
