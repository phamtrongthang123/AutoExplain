"""Small actual circuit-tracer fixture and portable attribution graph inspection."""
import torch
from .backends import require


def tiny_tracing_model(*, seed=7):
    """Random two-layer model and random transcoders, no downloads.

    Exercises the real attribution engine but cannot reveal learned semantic
    circuits. Use only in the dedicated transformers-4.x tracing environment.
    """
    from transformer_lens import HookedTransformerConfig
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    backend = Tokenizer(models.WordLevel({str(i): i for i in range(16)}, unk_token="0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="0", bos_token="1",
                                        eos_token="2", pad_token="0")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cfg = HookedTransformerConfig(n_layers=2, d_model=8, n_ctx=8, d_head=4, n_heads=2,
                                       d_mlp=16, d_vocab=16, act_fn="relu",
                                       normalization_type="LN", device="cpu")
        transcoders = {i: SingleLayerTranscoder(8, 12, torch.nn.ReLU(), i,
                       device=torch.device("cpu"), dtype=torch.float32) for i in range(2)}
        for transcoder in transcoders.values():
            with torch.no_grad():
                transcoder.W_enc.normal_(0, 0.1)
                transcoder.W_dec.normal_(0, 0.1)
        model = ReplacementModel.from_config(cfg, TranscoderSet(transcoders, "hook_resid_mid",
                                             "hook_mlp_out", scan_name="autoexplain-random-fixture"))
        # Tensor-token-only fixture: avoid TL's tokenizer reconstruction, which
        # otherwise attempts to locate/download a pretrained tokenizer.
        model.tokenizer = tokenizer
    return model.eval()


def top_graph_edges(graph, *, k=20):
    """Return largest absolute direct effects; upstream rows=target, cols=source.

    Rankings are not semantic labels or proof of circuit completeness. Excludes
    zero entries; retains sign. Caller must retain upstream node-type metadata.
    """
    matrix = graph.adjacency_matrix.detach().cpu()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.isfinite().all() or k < 1:
        raise ValueError("Expected finite square adjacency matrix and positive k")
    flat = matrix.flatten()
    indices = flat.abs().argsort(descending=True)[:k]
    n = matrix.shape[0]
    return [{"source": int(index % n), "target": int(index // n), "effect": float(flat[index])}
            for index in indices if flat[index] != 0]
