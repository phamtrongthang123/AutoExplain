"""A bounded, predefined real-model study; saves observations, not success claims."""
import time
import torch
from .pretrained import (encode_prompt,single_token_target,next_logits,margin,
                         record_residuals,patch_residual,layer_readouts,
                         embedding_attribution,random_matched_donor)

# These pairs are fixed before execution; all results, including failures, are retained.
CAPITAL_PAIRS = (
    ("The capital of France is", "The capital of Germany is", " Paris", " Berlin"),
    ("The capital of Italy is", "The capital of Spain is", " Rome", " Madrid"),
    ("The capital of Japan is", "The capital of China is", " Tokyo", " Beijing"),
)


def _top(logits,tokenizer,k=5):
    values,indices=logits.softmax(-1).topk(k)
    return [{"id":int(i),"text":tokenizer.decode([int(i)]),"probability":float(v)} for v,i in zip(values,indices)]


def run_pretrained_study(bundle, *, pairs=CAPITAL_PAIRS, layers=(0,8,16,24,34),
                         random_seeds=(7,11,19), include_gradients=True):
    """Three cloze contrasts, route-aware gradients, logit lens and residual patches.

    Patches compare donor, exact self-patch, and norm-matched random directions at
    the changed-country and last token. Final-layer last-token donor replacement
    is a trivial positive control, NOT circuit discovery. No parameter changes.
    No automatic prompt selection, hidden failed cases or semantic success gate.
    """
    if not 1 <= len(pairs) <= 6 or not 1 <= len(layers) <= 8 or not 1 <= len(random_seeds) <= 3:
        raise ValueError("Study exceeds predefined local budget")
    model,tokenizer=bundle.model,bundle.tokenizer
    started=time.perf_counter()
    report={"model":dict(bundle.metadata),"layers":list(layers),"random_seeds":list(random_seeds),
            "pairs":[],"status":"running","limitations":[
                "Three predefined fact-completion contrasts are examples, not a benchmark.",
                "No semantic concept discovery or J-lens fitting is claimed by plain logit-lens readouts.",
                "Embedding gradient routes are conditional sensitivities, not completeness-certified attributions.",
                "Residual patches leave token IDs, per-layer token embeddings and shared KV routes otherwise intact.",
                "Final-layer last-token donor patch copies the answer readout: it is a positive control only.",
                "Norm-matched random directions are sampled controls, not a statistical significance test.",
                "Recovery ratios are not probabilities, may exceed one and depend on a nonzero baseline gap.",
                "BF16 is unquantized here, but finite precision can affect small differences."]}
    for clean_text,recipient_text,target_text,foil_text in pairs:
        clean_ids=encode_prompt(bundle,clean_text)
        recipient_ids=encode_prompt(bundle,recipient_text)
        if clean_ids.shape != recipient_ids.shape:
            raise ValueError("Prompts must have exactly aligned token lengths")
        differing=(clean_ids[0]!=recipient_ids[0]).nonzero().flatten().tolist()
        if len(differing)!=1:
            raise ValueError("Each predefined contrast must change exactly one token")
        positions={"changed_token":differing[0],"last_token":clean_ids.shape[1]-1}
        if positions['changed_token']==positions['last_token']:
            raise ValueError("Changed token must precede the answer readout position")
        target=single_token_target(tokenizer,target_text)
        foil=single_token_target(tokenizer,foil_text)
        with record_residuals(model,layers) as donor:
            clean_logits=next_logits(model,clean_ids)
        with record_residuals(model,layers) as recipient:
            recipient_logits=next_logits(model,recipient_ids)
        clean_margin=margin(clean_logits,target,foil)
        recipient_margin=margin(recipient_logits,target,foil)
        gap=clean_margin-recipient_margin
        result={"clean_prompt":clean_text,"recipient_prompt":recipient_text,
                "target":target_text,"foil":foil_text,"target_id":target,"foil_id":foil,
                "token_labels":tokenizer.convert_ids_to_tokens(clean_ids[0].tolist()),
                "positions":positions,"clean_margin":clean_margin,"recipient_margin":recipient_margin,
                "clean_target_is_top1":int(clean_logits.argmax())==target,
                "recipient_foil_is_top1":int(recipient_logits.argmax())==foil,
                "clean_top5":_top(clean_logits,tokenizer),"recipient_top5":_top(recipient_logits,tokenizer),
                "readouts":layer_readouts(model,donor,target,foil),"patches":[],"self_controls":[]}
        if include_gradients:
            result['embedding_attribution']=embedding_attribution(model,clean_ids,target,foil)
        for layer in layers:
            with patch_residual(model,layer,recipient[layer],positions=list(positions.values())):
                self_logits=next_logits(model,recipient_ids)
            self_error=float((self_logits-recipient_logits).abs().max())
            result['self_controls'].append({"layer":layer,"max_logit_error":self_error})
            if self_error > 1e-3:
                raise RuntimeError("Self-patch changed logits; reject nondeterministic/misaligned control")
            for name,position in positions.items():
                with patch_residual(model,layer,donor[layer],positions=[position]):
                    patched=next_logits(model,recipient_ids)
                patched_margin=margin(patched,target,foil)
                random_margins=[]
                for seed in random_seeds:
                    random_donor=random_matched_donor(recipient[layer],donor[layer],positions=[position],seed=seed)
                    with patch_residual(model,layer,random_donor,positions=[position]):
                        random_logits=next_logits(model,recipient_ids)
                    random_margins.append(margin(random_logits,target,foil))
                result['patches'].append({"layer":layer,"position_name":name,"position":position,
                    "margin":patched_margin,"top_token_id":int(patched.argmax()),
                    "top_token":tokenizer.decode([int(patched.argmax())]),
                    "margin_change":patched_margin-recipient_margin,
                    "recovery_ratio":None if abs(gap)<1e-6 else (patched_margin-recipient_margin)/gap,
                    "random_margins":random_margins,
                    "trivial_final_readout_control":layer==len(model.model.layers)-1 and name=='last_token'})
        if len(model.model.layers)-1 in donor:
            final_readout=next(x for x in result['readouts'] if x['layer']==len(model.model.layers)-1)
            result['final_readout_margin_error']=abs(final_readout['margin']-clean_margin)
            if result['final_readout_margin_error']>1e-3:
                raise RuntimeError("Final logit-lens readout disagrees with actual output")
        report['pairs'].append(result)
        del donor,recipient
    report.update(status='completed',experiment_seconds=time.perf_counter()-started)
    if torch.cuda.is_available() and model.get_input_embeddings().weight.device.type=='cuda':
        device=model.get_input_embeddings().weight.device
        report['gpu_peak_allocated_gib']=torch.cuda.max_memory_allocated(device)/1024**3
        report['gpu_peak_reserved_gib']=torch.cuda.max_memory_reserved(device)/1024**3
        report['gpu_name']=torch.cuda.get_device_name(device)
    return report
