"""No checkpoint downloads or GPU requirement: exact small native Gemma4 fixtures."""
import pytest
import torch

transformers=pytest.importorskip('transformers')
if not hasattr(transformers, 'Gemma4ForCausalLM'):
    pytest.skip('Native Gemma4 requires recent Transformers',allow_module_level=True)

from autoexplain.pretrained import (CPUEmbeddingBridge,place_gemma4,encode_prompt,
    single_token_target,PretrainedBundle,next_logits,record_residuals,patch_residual,
    layer_readouts,embedding_attribution,random_matched_donor,load_gemma4)


def tiny_model():
    from transformers import Gemma4ForCausalLM,Gemma4TextConfig
    torch.manual_seed(7)
    config=Gemma4TextConfig(vocab_size=32,vocab_size_per_layer_input=32,
        hidden_size=16,hidden_size_per_layer_input=4,intermediate_size=32,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,
        head_dim=8,global_head_dim=8,num_kv_shared_layers=0,
        layer_types=['sliding_attention','sliding_attention','full_attention'],
        max_position_embeddings=32,sliding_window=8,use_double_wide_mlp=False)
    config._attn_implementation='eager'
    return Gemma4ForCausalLM(config).eval().requires_grad_(False)


def test_cpu_bridge_exact_and_main_ple_gradients():
    model=tiny_model(); ids=torch.tensor([[2,3,4,5]])
    original=next_logits(model,ids)
    place_gemma4(model,decoder_device='cpu')
    assert isinstance(model.model.embed_tokens_per_layer,CPUEmbeddingBridge)
    assert torch.equal(original,next_logits(model,ids))
    attribution=embedding_attribution(model,ids,6,7)
    assert set(attribution)=={'main_embedding','per_layer_embedding'}
    for values in attribution.values():
        assert len(values['gradient_l2'])==4 and all(x>=0 for x in values['gradient_l2'])
    assert all(p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks for m in model.modules())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA placement equivalence test')
def test_cuda_bridge_matches_native_full_gpu():
    model=tiny_model().to('cuda')
    ids=torch.tensor([[2,3,4,5]],device='cuda')
    native=next_logits(model,ids)
    place_gemma4(model,decoder_device='cuda')
    split=next_logits(model,ids)
    assert model.model.embed_tokens_per_layer.weight.device.type=='cpu'
    assert torch.allclose(native,split,atol=1e-6,rtol=1e-5)
    gradients=embedding_attribution(model,ids,6,7)
    assert all(len(route['gradient_l2'])==4 for route in gradients.values())


def test_patch_controls_and_final_readout():
    model=tiny_model(); ids=torch.tensor([[2,3,4,5]])
    with record_residuals(model,[0,2]) as captured:
        baseline=next_logits(model,ids)
    with patch_residual(model,0,captured[0],positions=[1]):
        assert torch.equal(next_logits(model,ids),baseline)
    rows=layer_readouts(model,captured,6,7)
    assert rows[-1]['margin']==pytest.approx(float(baseline[6]-baseline[7]),abs=1e-6)
    donor=captured[0]+.1
    random=random_matched_donor(captured[0],donor,positions=[1],seed=7)
    assert torch.equal(random[:,0],captured[0][:,0])
    assert torch.allclose((random[:,1]-captured[0][:,1]).norm(dim=-1),
                          (donor[:,1]-captured[0][:,1]).norm(dim=-1),atol=1e-6)
    with pytest.raises(RuntimeError):
        with patch_residual(model,0,captured[0],positions=[1]):raise RuntimeError('failure')
    assert all(not m._forward_hooks for m in model.modules())


def test_native_multimodal_checkpoint_text_conversion(tmp_path):
    from transformers import Gemma4Config,Gemma4ForConditionalGeneration,Gemma4ForCausalLM
    tiny=tiny_model()
    full=Gemma4ForConditionalGeneration(Gemma4Config(text_config=tiny.config,vision_config=None,audio_config=None)).eval()
    ids=torch.tensor([[2,3,4]])
    with torch.no_grad(): expected=full(input_ids=ids,use_cache=False).logits
    full.save_pretrained(tmp_path)
    loaded,info=Gemma4ForCausalLM.from_pretrained(tmp_path,local_files_only=True,output_loading_info=True,
        key_mapping={r"^model\.language_model\.": "model."})
    assert not info['missing_keys'] and not info['mismatched_keys']
    with torch.no_grad():actual=loaded(input_ids=ids,use_cache=False).logits
    assert torch.allclose(actual,expected,atol=1e-6)


def test_study_preserves_controls():
    from autoexplain.pretrained_study import run_pretrained_study
    class Tokenizer:
        def encode(self,text,add_special_tokens=True):
            return {'clean':[2,3,4], 'recipient':[2,5,4], ' target':[6], ' foil':[7]}[text]
        def decode(self,ids):return ','.join(map(str,ids))
        def convert_ids_to_tokens(self,ids):return list(map(str,ids))
    bundle=PretrainedBundle(tiny_model(),Tokenizer(),{'fixture':True})
    report=run_pretrained_study(bundle,pairs=[('clean','recipient',' target',' foil')],layers=[0,2],random_seeds=[7])
    assert report['status']=='completed' and len(report['pairs'][0]['patches'])==4
    assert all(row['max_logit_error']==0 for row in report['pairs'][0]['self_controls'])
    assert report['pairs'][0]['patches'][-1]['trivial_final_readout_control']
    assert all(not m._forward_hooks for m in bundle.model.modules())


def test_loading_guard_no_download():
    with pytest.raises(RuntimeError,match='host memory'):
        load_gemma4(decoder_device='cpu',min_host_available_gib=100000)


def test_alignment_token_contracts():
    class Tokenizer:
        def encode(self,text,add_special_tokens=True):return [2,3] if text=='long' else [4]
    with pytest.raises(ValueError):single_token_target(Tokenizer(),'long')
    assert single_token_target(Tokenizer(),'single')==4
    bundle=PretrainedBundle(tiny_model(),Tokenizer(),{})
    with pytest.raises(ValueError):encode_prompt(bundle,'long',max_tokens=1)
