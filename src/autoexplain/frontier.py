"""Explicit optional adapters to public frontier-lab research code.

No model weights are downloaded by these functions. Caller-supplied model objects
and tiny random models test mechanics only, not paper reproduction.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import torch
from torch import nn
from .backends import require


class TinyLensModel(nn.Module):
    """Offline LensModel protocol fixture: tiny causal residual stack + character IDs.

    Not a pretrained language model. Token-wise residual MLPs provide a simple
    causal model for reference J-lens estimation and readout checks.
    """
    def __init__(self, width=8, layers=3, vocabulary=32):
        super().__init__()
        self.d_model, self.n_layers, self.vocabulary = width, layers, vocabulary
        self.embedding = nn.Embedding(vocabulary, width)
        self.layers = nn.ModuleList([ResidualMLP(width) for _ in range(layers)])
        self.head = nn.Linear(width, vocabulary, bias=False)
        self.tokenizer = NumericTokenizer()

    def encode(self, text, *, max_length=32):
        return torch.tensor([[ord(c) % self.vocabulary for c in text[:max_length]]],
                            device=self.embedding.weight.device)

    def decode(self, token_ids):
        return " ".join(str(int(i)) for i in token_ids)

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.unembed(h)

    def unembed(self, residual):
        return self.head(residual)


class NumericTokenizer:
    def decode(self, token_ids):
        return " ".join(str(int(i)) for i in token_ids)


class ResidualMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return x + self.linear(x).tanh()


def tiny_hf_causal_lm(*, seed=7):
    """Actual Hugging Face GPT-2 architecture with random tiny weights and numeric tokenizer."""
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    backend = Tokenizer(models.WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="0", bos_token="1",
                                        eos_token="2", pad_token="0")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=32, vocab_size=32,
                            bos_token_id=1, eos_token_id=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    return model, tokenizer


def tiny_sparse_circuit_model(*, width=8, vocabulary=16, seed=7):
    """Instantiate the actual OpenAI circuit_sparsity GPT, no weight downloads.

    Uses ReLU activations with activation sparsity. Random weights are NOT a
    learned sparse circuit and this does not reproduce the released models.
    """
    backend = require("circuit_sparsity.inference.gpt", "openai")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = backend.GPTConfig(block_size=16, vocab_size=vocabulary, n_layer=2,
                                   n_head=1, d_model=width, d_mlp=width*2,
                                   activation_type="relu", afrac=0.5,
                                   grad_checkpointing=False, flash=False)
        return backend.GPT(config).eval()


def inspect_sparse_circuit(model, tokens):
    hooks = require("circuit_sparsity.inference.hook_utils", "openai")
    with torch.no_grad(), hooks.hook_recorder() as records:
        logits, _, _ = model(tokens)
    return {"logits": logits, "activations": dict(records),
            "parameter_zero_fraction": sum((p == 0).sum().item() for p in model.parameters()) /
                                       sum(p.numel() for p in model.parameters())}


def run_goodfire_nano(target, module_components, train_batches, eval_batches, *, steps=2):
    """Run vendored upstream VPD training on a caller-owned tiny CPU LM.

    Mutates/replaces target linear modules and freezes original weights. This
    deliberately uses the upstream nano L_p objective, not current JAX smooth-L0.
    No W&B initialization/logging. Fixed tiny settings are a mechanics demo, not
    a scientifically converged decomposition. Requires torch>=2.4 + wandb import.
    """
    if not 1 <= steps <= 20:
        raise ValueError("Local nano demo limited to 1–20 steps")
    if torch.cuda.is_available():
        raise ValueError("Run nano demo with CUDA_VISIBLE_DEVICES='' to keep it CPU-only")
    if any(p.device.type != 'cpu' for p in target.parameters()):
        raise ValueError("CPU model required")
    require("wandb", "goodfire-local")
    from ._vendor.goodfire_nano import Config, decompose
    first = next(train_batches)
    if first.ndim != 2 or not 1 <= first.shape[0] <= 4 or not 2 <= first.shape[1] <= 16:
        raise ValueError("Need tiny [B,T] token batches, B<=4, 2<=T<=16")
    from itertools import chain
    cfg = Config(C_per_module=module_components, n_steps=steps,
                 batch_size=first.shape[0], seq_len=first.shape[1],
                 faithfulness_warmup_steps=2, ci_d_model=8, ci_n_blocks=1,
                 ci_n_heads=1, ci_mlp_hidden=16, ppgd_inner_steps=1,
                 pgd_eval_n_steps=1, eval_batch_size=first.shape[0],
                 eval_freq=20, slow_eval_freq=20, slow_eval_on_first_step=False,
                 use_wandb=False, log_every=1)
    decompose(target, cfg, iter(chain([first], train_batches)), eval_batches)
    return target


def trace_circuit(replacement_model, prompt, *, max_nodes=64, batch_size=1):
    """Call circuit-tracer on an ALREADY loaded compatible replacement model.

    Install in the separate tracing environment (transformers 4.x), not the
    J-lens environment (5.x). No automatic downloads or model construction here.
    """
    if not 1 <= max_nodes <= 1024 or not 1 <= batch_size <= 8:
        raise ValueError("Requested graph exceeds local adapter budget")
    tracer = require("circuit_tracer", "tracing")
    return tracer.attribute(prompt=prompt, model=replacement_model, max_n_logits=1,
                            max_feature_nodes=max_nodes, batch_size=batch_size, offload=None)


@dataclass
class GoodfireRemote:
    """Optional hosted Ember client. Explicit key and opt-in; never used in tutorials.

    Current service enrollment/availability is unverified and SDK is archived.
    Only explicitly passed text is sent. Network failures propagate; no retries here.
    """
    api_key: str = field(repr=False)
    allow_remote: bool = False

    def search(self, query, *, model, top_k=5):
        if not self.allow_remote:
            raise PermissionError("Hosted access requires allow_remote=True")
        if not self.api_key or not 1 <= top_k <= 50:
            raise ValueError("Need key and bounded top_k")
        sdk = require("goodfire", "goodfire-remote")
        client = sdk.Client(api_key=self.api_key)
        return client.features.search(query, model=sdk.Variant(model), top_k=top_k)
