import numpy as np
import pytest
import torch
from torch import nn
from autoexplain.backends import captum_attribute, attribution_infidelity, attribution_sensitivity
from autoexplain.diffusion import diffusion_intervention, sample_diffusers
from autoexplain.features import load_saelens_local,load_saelens_release
from autoexplain.tabular_tools import permutation_importance, partial_dependence_ice, nearest_counterfactual
from autoexplain.concepts import fit_cav,nmf_features
from autoexplain.representations import parameter_randomization_check
from autoexplain.catalog import list_tools


def test_infidelity_linear_exact():
    pytest.importorskip('captum')
    model=nn.Linear(3,1,bias=False)
    model.weight.data.copy_(torch.tensor([[1.,2.,3.]]))
    x=torch.ones(2,3)
    attrs=model.weight.detach().expand_as(x)
    def perturb(x):
        delta=torch.ones_like(x)*.01
        return delta,x-delta
    score=attribution_infidelity(model,x,attrs,perturb,target=0,samples=4)
    assert torch.all(score < 1e-10)
    explain=lambda x:captum_attribute(model,x,method='saliency',target=0)
    assert torch.all(attribution_sensitivity(explain,x,samples=2) < 1e-6)


def test_diffusion_branch_mask_and_cleanup():
    model=nn.Sequential(nn.Identity())
    x=torch.zeros(2,3,4,4)
    mask=torch.zeros(1,1,4,4);mask[:,:,:2]=1
    with diffusion_intervention(model,'0',torch.tensor([1.,0.,0.]),batch_indices=[1],spatial_mask=mask):
        y=model(x)
    assert y[0].count_nonzero()==0 and y[1,0].sum()==8
    assert not model[0]._forward_hooks
    with pytest.raises(ValueError):
        with diffusion_intervention(model,'0',torch.ones(3),batch_indices=[2]): model(x)
    assert not model[0]._forward_hooks


def test_real_diffusers_matched_noise():
    pytest.importorskip('diffusers')
    from diffusers import UNet2DModel, DDPMScheduler
    torch.set_num_threads(2)
    net=UNet2DModel(sample_size=8,in_channels=1,out_channels=1,layers_per_block=1,
        block_out_channels=(8,),down_block_types=('DownBlock2D',),up_block_types=('UpBlock2D',),
        norm_num_groups=4,add_attention=False)
    noise=torch.randn(1,1,8,8)
    def scheduler():return DDPMScheduler(num_train_timesteps=20)
    baseline=sample_diffusers(net,scheduler(),noise,steps=3,seed=7)
    zero=sample_diffusers(net,scheduler(),noise,steps=3,seed=7,active_steps=[0,1,2],
        intervention=lambda:diffusion_intervention(net,'conv_in',torch.ones(8),strength=0))
    assert torch.equal(baseline,zero)
    changed=sample_diffusers(net,scheduler(),noise,steps=3,seed=7,active_steps=[1],
        intervention=lambda:diffusion_intervention(net,'conv_in',torch.ones(8),strength=.2))
    assert changed.isfinite().all() and not torch.equal(changed,baseline)


def test_saelens_round_trip(tmp_path):
    pytest.importorskip('sae_lens')
    from sae_lens.saes.jumprelu_sae import JumpReLUSAE,JumpReLUSAEConfig
    sae=JumpReLUSAE(JumpReLUSAEConfig(d_in=4,d_sae=8))
    sae.save_model(tmp_path/'sae')
    loaded=load_saelens_local(tmp_path/'sae')
    x=torch.randn(3,4)
    assert torch.equal(sae.encode(x),loaded.encode(x)[0])
    with pytest.raises(PermissionError):load_saelens_release('not-downloaded','id')


def test_tabular_and_concepts():
    from sklearn.tree import DecisionTreeClassifier
    x=np.random.default_rng(7).normal(size=(60,3)); y=(x[:,0]>0).astype(int)
    model=DecisionTreeClassifier(max_depth=2).fit(x,y)
    importance=permutation_importance(model,x,y,repeats=2)
    assert importance.importances_mean.shape==(3,)
    pdp=partial_dependence_ice(model,x,[0],grid_resolution=5)
    assert pdp.individual.shape[-1]==5
    cf=nearest_counterfactual(model,x[0],x,target=1-int(y[0]))
    assert cf is not None and model.predict(cf['counterfactual'][None])[0] != y[0]
    tx=torch.from_numpy(x).float()
    cav=fit_cav(tx[torch.tensor(y)==1],tx[torch.tensor(y)==0])
    assert cav['direction'].norm()==pytest.approx(1.)
    result=nmf_features(tx.abs(),components=2)
    assert result['codes'].shape==(60,2)


def test_catalog_and_randomization():
    names=[t.name for t in list_tools()]
    assert len(names)==len(set(names)) and len(names)>50
    model=nn.Linear(3,2)
    def explain(m,x):return m.weight[0].expand_as(x)
    score=parameter_randomization_check(lambda:nn.Linear(3,2),model,torch.ones(2,3),explain)
    assert score.shape==(2,) and score.isfinite().all()
