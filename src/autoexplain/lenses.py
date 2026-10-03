"""Jacobian transport and adapters to Anthropic's reference implementation.

J-space is the verbalizable representation space, not an independent model SDK.
"""
from dataclasses import dataclass
import torch
from .backends import require


@dataclass
class LocalJacobianLens:
    matrix: torch.Tensor
    samples: int

    def transport(self, activations):
        return activations @ self.matrix.to(activations).T

    def readout(self, activations, unembed):
        return unembed(self.transport(activations))


def fit_local_jacobian(tail, sequences, *, skip_first=0, max_width=64):
    """Small exact autograd estimator for a caller-defined causal residual tail.

    Inputs are [T,D] sequences; tail returns [T,D]. For each output dimension,
    sum over valid target positions then average gradients over valid source
    positions. Excludes the final token, following the upstream reduction. This
    does not fit an entire arbitrary transformer or reproduce paper findings.
    Caller must ensure tail is causal and deterministic (eval, no dropout).
    """
    if not sequences or type(skip_first) is not int or skip_first < 0:
        raise ValueError("Need sequences and nonnegative skip_first")
    matrices = []
    width = None
    with torch.enable_grad():
        for sample in sequences:
            if sample.ndim != 2 or not sample.is_floating_point() or not sample.isfinite().all():
                raise ValueError("Expected finite floating [T,D] sequences")
            if not 1 <= sample.shape[1] <= max_width or sample.shape[0] <= skip_first + 1:
                raise ValueError("Sequence too short or width exceeds local Jacobian budget")
            if width is not None and width != sample.shape[1]:
                raise ValueError("Feature widths must agree")
            width = sample.shape[1]
            x = sample.detach().clone().requires_grad_(True)
            output = tail(x)
            if output.shape != x.shape:
                raise ValueError("Tail must preserve [T,D] shape")
            rows = []
            for dim in range(width):
                grad = torch.autograd.grad(output[skip_first:-1, dim].sum(), x,
                                           retain_graph=dim < width - 1)[0]
                rows.append(grad[skip_first:-1].mean(0))
            matrices.append(torch.stack(rows).detach())
    matrix = torch.stack(matrices).mean(0)
    if not matrix.isfinite().all():
        raise ValueError("Nonfinite Jacobian")
    return LocalJacobianLens(matrix, len(matrices))


def fit_reference_jlens(lens_model, prompts, *, source_layers, max_seq_len=32,
                        skip_first=0, dim_batch=2):
    """Run the actual upstream jlens.fit on an explicitly prepared LensModel.

    Tokenization/model loading remain caller-owned. No automatic downloads.
    Using jlens.from_hf separately mutates/freezes the supplied model; consult
    upstream documentation. Small prompt sets validate mechanics, not semantics.
    """
    if not prompts or not source_layers or max_seq_len < skip_first + 2 or dim_batch < 1:
        raise ValueError("Invalid fitting budget")
    jlens = require("jlens", "jlens")
    return jlens.fit(lens_model, prompts=list(prompts), source_layers=list(source_layers),
                     max_seq_len=max_seq_len, skip_first=skip_first, dim_batch=dim_batch,
                     checkpoint_path=None)


def read_reference_jlens(lens, lens_model, prompt, *, positions=None, max_seq_len=32):
    """Return matched J-lens and plain logit-lens readouts from the upstream API."""
    j, actual, ids = lens.apply(lens_model, prompt, positions=positions, max_seq_len=max_seq_len)
    plain, _, _ = lens.apply(lens_model, prompt, positions=positions, max_seq_len=max_seq_len,
                             use_jacobian=False)
    return {"jacobian_lens": j, "logit_lens": plain, "actual": actual, "input_ids": ids}
