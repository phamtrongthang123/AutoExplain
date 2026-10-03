import itertools
import pytest
import torch
from torch import nn
from autoexplain.features import SAEAdapter, steer_sae, feature_examples
from autoexplain.lenses import fit_local_jacobian, fit_reference_jlens, read_reference_jlens
from autoexplain.frontier import TinyLensModel, tiny_sparse_circuit_model, inspect_sparse_circuit, run_goodfire_nano, GoodfireRemote
from autoexplain.representations import linear_cka, attention_rollout, concept_sensitivity, ablate_features, logit_lens, ridge_probe


def test_linear_jacobian_exact():
    w = torch.tensor([[1.,2.],[-1.,3.]])
    lens = fit_local_jacobian(lambda x: x @ w.T, [torch.randn(5,2),torch.randn(6,2)])
    assert torch.allclose(lens.matrix, w)
    x = torch.randn(2,2)
    assert torch.allclose(lens.transport(x), x @ w.T)


def test_reference_jlens():
    pytest.importorskip('jlens')
    torch.set_num_threads(2)
    model = TinyLensModel().eval()
    lens = fit_reference_jlens(model, ['abcdefghi','jklmnopqr'], source_layers=[0,1], max_seq_len=12)
    result = read_reference_jlens(lens, model, 'abcdef', positions=[2])
    assert lens.n_prompts == 2
    assert result['actual'].shape == (1,32)
    assert all(x.isfinite().all() for x in result['jacobian_lens'].values())
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize('backend',['openai','saelens'])
def test_actual_sae_adapters(backend):
    if backend == 'openai':
        from autoexplain._vendor.openai_sae import Autoencoder
        sae = Autoencoder(n_latents=12,n_inputs=4)
    else:
        pytest.importorskip('sae_lens')
        from sae_lens.saes.standard_sae import StandardSAE,StandardSAEConfig
        sae = StandardSAE(StandardSAEConfig(d_in=4,d_sae=12))
    adapter = SAEAdapter(sae,backend=backend)
    x = torch.randn(2,3,4)
    result = adapter.inspect(x,top_k=2)
    assert result['features'].shape == (2,3,12)
    assert torch.allclose(adapter.edit(x,feature=0,scale=1),x,atol=1e-6)
    model = nn.Sequential(nn.Identity())
    with steer_sae(model,'0',adapter,feature=0,scale=0):
        assert model(x).shape == x.shape
    assert not model[0]._forward_hooks
    assert feature_examples(result['features'],0,k=2)['scores'].shape == (2,)


def test_sparse_circuits_actual_model():
    pytest.importorskip('circuit_sparsity')
    model = tiny_sparse_circuit_model()
    result = inspect_sparse_circuit(model,torch.tensor([[1,2,3,4]]))
    assert result['logits'].shape[-1] == 16
    assert result['activations'] and result['logits'].isfinite().all()


def test_goodfire_actual_nano_training():
    pytest.importorskip('wandb')
    torch.set_num_threads(2)
    model = nn.Sequential(nn.Embedding(16,4),nn.Linear(4,16))
    data = itertools.cycle([torch.tensor([[1,2,3,4],[2,3,4,5]])])
    trained = run_goodfire_nano(model,{'1':4},data,data,steps=2)
    assert hasattr(trained[1],'V')
    assert trained(torch.tensor([[1,2]])).isfinite().all()


def test_representations_and_ablation():
    x = torch.randn(20,3)
    assert linear_cka(x,x) == pytest.approx(1.)
    assert linear_cka(x,2*x+3) == pytest.approx(1.)
    with pytest.raises(ValueError): linear_cka(torch.ones(3,2),torch.ones(3,2))
    assert concept_sensitivity(torch.tensor([[1.,0.],[-1.,0.]]),torch.tensor([1.,0.])) == 0.5
    attention = torch.ones(1,2,4,4)/4
    assert torch.allclose(attention_rollout([attention,attention]).sum(-1),torch.ones(1,4))
    model=nn.Sequential(nn.Identity())
    with ablate_features(model,'0',[1]):
        assert torch.equal(model(torch.ones(1,3)),torch.tensor([[1.,0.,1.]]))
    assert not model[0]._forward_hooks
    assert logit_lens([x],nn.Linear(3,2))[0].shape==(20,2)
    _,pred=ridge_probe(x,(x[:,0]>0).long(),x[:2])
    assert len(pred)==2


def test_remote_requires_opt_in():
    with pytest.raises(PermissionError): GoodfireRemote('not-a-real-key').search('test',model='test')
    assert 'not-a-real-key' not in repr(GoodfireRemote('not-a-real-key'))


def test_remote_transport_contract_mock(monkeypatch):
    from types import SimpleNamespace
    import autoexplain.frontier as frontier
    calls = []
    def search(query, model, top_k):
        calls.append((query, model, top_k))
        return ['mock feature']
    backend = SimpleNamespace(Client=lambda api_key: SimpleNamespace(features=SimpleNamespace(search=search)),
                              Variant=lambda model: ('variant', model))
    monkeypatch.setattr(frontier, 'require', lambda module, extra: backend)
    result = GoodfireRemote('not-a-real-key', allow_remote=True).search('concept', model='model-id', top_k=3)
    assert result == ['mock feature'] and calls == [('concept', ('variant', 'model-id'), 3)]
