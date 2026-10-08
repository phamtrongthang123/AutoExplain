"""Actual pretrained Gemma4 Jacobian lens, with held-out readouts and a scrambled control."""
import time
import torch
from .pretrained import encode_prompt
from .lenses import fit_reference_jlens
from .pretrained_study import CAPITAL_PAIRS

FIT_PROMPTS = (
    'A small bird landed on a branch near the window.',
    'The student opened a book and began reading quietly.',
    'Several people walked through the park after the rain.',
    'We placed the wooden box on the table beside the door.',
)


def _rank(logits, token):
    """Competition rank: 1 + number of strictly larger scores (ties share rank)."""
    return int((logits > logits[token]).sum()) + 1


def _top(logits, tokenizer, k=5):
    values, indices=logits.topk(k)
    return [{"id":int(i),"text":tokenizer.decode([int(i)]),"logit":float(v)} for v,i in zip(values,indices)]


def fit_pretrained_jlens(bundle, *, prompts=FIT_PROMPTS, layer=8, dim_batch=8, seed=11):
    """Four generic fit prompts, six disjoint evaluation prompts, one fitted layer.

    Calls Anthropic's actual estimator, not a toy substitute. The scrambled lens
    permutes input-coordinate columns of the fitted matrix, preserving matrix
    norms/singular values while disrupting alignment. One scrambled control and
    six illustrative examples are NOT a significance test or paper reproduction.
    """
    import jlens
    if not 1 <= len(prompts) <= 8 or not 1 <= dim_batch <= 8:
        raise ValueError("This demonstration is bounded to eight prompts and dimension batch eight")
    if not 0 <= layer < len(bundle.model.model.layers)-1:
        raise ValueError("Fit a layer strictly before the final layer")
    evaluation_prompts=[text for pair in CAPITAL_PAIRS for text in pair[:2]]
    if set(prompts).intersection(evaluation_prompts):
        raise ValueError("Fitting and evaluation prompts must be disjoint")
    adapted=jlens.from_hf(bundle.model,bundle.tokenizer,force_bos=False)
    started=time.perf_counter()
    lens=fit_reference_jlens(adapted,prompts,source_layers=[layer],max_seq_len=24,
                             skip_first=1,dim_batch=dim_batch)
    if lens.n_prompts != len(prompts):
        raise RuntimeError("Upstream skipped a fitting prompt; revise the corpus explicitly")
    fit_seconds=time.perf_counter()-started
    permutation=torch.randperm(lens.d_model,generator=torch.Generator().manual_seed(seed))
    scrambled=jlens.JacobianLens({layer:lens.jacobians[layer][:,permutation]},
                                n_prompts=lens.n_prompts,d_model=lens.d_model)
    report={"model":dict(bundle.metadata),"layer":layer,"fit_prompts":list(prompts),
            "fit_prompt_count":lens.n_prompts,"fit_seconds":fit_seconds,
            "estimator":{"max_seq_len":24,"skip_first":1,"dim_batch":dim_batch,
                         "reduction":"sum valid target positions, average valid source positions, then mean prompts"},
            "scramble_seed":seed,"transport_frobenius_norm":float(lens.jacobians[layer].norm()),
            "rank_definition":"1 + number of strictly larger vocabulary scores; ties share rank",
            "readouts":[],"limitations":[
                "Only four generic prompts estimate one layer; fit stability is not established.",
                "Input-country recovery is a readability diagnostic, not next-token accuracy or reasoning validation.",
                "One scrambled-matrix control and six examples are not a statistical significance test.",
                "J-lens reads dispositions under an average linear transport, not literal internal thoughts.",
                "Other token-identity/PLE and shared-KV paths are held fixed by the source-activation derivative.",
                "The language model is pretrained; the fitted lens is an illustrative small-corpus estimate."]}
    for text in evaluation_prompts:
        ids=encode_prompt(bundle,text)
        if ids.shape[1]!=6:
            raise ValueError("The fixed evaluation template must tokenize to six aligned tokens")
        country_position,last_position=4,5
        country_id=int(ids[0,country_position])
        j,actual,_=lens.apply(adapted,text,positions=[country_position,last_position],max_seq_len=24)
        plain,_,_=lens.apply(adapted,text,positions=[country_position,last_position],max_seq_len=24,use_jacobian=False)
        random,_,_=scrambled.apply(adapted,text,positions=[country_position,last_position],max_seq_len=24)
        row={"prompt":text,"country_token_id":country_id,"country_token":bundle.tokenizer.decode([country_id]),
             "positions":{"country":country_position,"last":last_position},
             "country_token_ranks":{},"country_readouts":{},"last_readouts":{},
             "actual_next_token_top5":_top(actual[1],bundle.tokenizer)}
        for name,values in [('jacobian_lens',j[layer]),('logit_lens',plain[layer]),('scrambled_lens',random[layer])]:
            if not values.isfinite().all():raise ValueError("Nonfinite lens readout")
            row['country_token_ranks'][name]=_rank(values[0],country_id)
            row['country_readouts'][name]=_top(values[0],bundle.tokenizer)
            row['last_readouts'][name]=_top(values[1],bundle.tokenizer)
        report['readouts'].append(row)
    report.update(status='completed',total_fit_and_readout_seconds=time.perf_counter()-started)
    return lens,report
