"""Small deterministic contracts; real estimator evidence lives in notebook 15."""
from types import SimpleNamespace
import sys
import pytest
import torch
from autoexplain.pretrained_jlens import _rank, fit_pretrained_jlens, FIT_PROMPTS
from autoexplain.pretrained_study import CAPITAL_PAIRS


def test_rank_uses_competition_ties():
    assert _rank(torch.tensor([2., 2., 1., 3.]), 0) == 2
    assert _rank(torch.tensor([2., 2., 1., 3.]), 2) == 4


@pytest.mark.parametrize('kwargs', [
    {'prompts': []}, {'prompts': ['x'] * 9}, {'dim_batch': 0},
    {'dim_batch': 9}, {'layer': 2}, {'layer': -1},
    {'prompts': [CAPITAL_PAIRS[0][0]]},
])
def test_invalid_fit_rejected_before_upstream(monkeypatch, kwargs):
    monkeypatch.setitem(sys.modules, 'jlens', SimpleNamespace())
    bundle = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=[0, 1, 2])))
    with pytest.raises(ValueError):
        fit_pretrained_jlens(bundle, **({'layer': 1} | kwargs))


def test_fit_and_evaluation_corpora_disjoint():
    evaluation = {text for pair in CAPITAL_PAIRS for text in pair[:2]}
    assert not set(FIT_PROMPTS) & evaluation


def test_skipped_fit_prompt_rejected(monkeypatch):
    import autoexplain.pretrained_jlens as module
    monkeypatch.setitem(sys.modules, 'jlens', SimpleNamespace(from_hf=lambda *a, **kw: object()))
    monkeypatch.setattr(module, 'fit_reference_jlens', lambda *a, **kw: SimpleNamespace(n_prompts=3))
    bundle = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=[0, 1, 2])), tokenizer=object())
    with pytest.raises(RuntimeError, match='skipped'):
        fit_pretrained_jlens(bundle, layer=1)
