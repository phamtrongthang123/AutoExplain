"""Discover callable tools; listed is not equivalent to scientifically validated."""
from dataclasses import asdict, dataclass
import argparse
import importlib.util
import json
from .backends import CAPTUM_METHODS, CAM_METHODS, LAYER_METHODS


@dataclass(frozen=True)
class Tool:
    name: str
    family: str
    entrypoint: str
    extra: str
    scope: str


def list_tools(family=None):
    tools = []
    for name in CAPTUM_METHODS:
        tools.append(Tool('captum.'+name, 'attribution', 'autoexplain.backends.captum_attribute',
                          'attribution', 'Local integration tested on small explicit score models'))
    for name in CAM_METHODS:
        tools.append(Tool('cam.'+name, 'vision', 'autoexplain.backends.cam_attribute',
                          'vision', 'Local integration tested on small CNN; EigenCAM is class-independent'))
    for name in LAYER_METHODS:
        tools.append(Tool('captum.'+name, 'attribution', 'autoexplain.backends.captum_layer',
                          'attribution', 'Explicit named module and scalar/class target'))
    for name in ('tree','permutation','exact','linear'):
        tools.append(Tool('shap.'+name,'tabular','autoexplain.backends.shap_explain','tabular',
                          'Native SHAP Explanation; explicit background where required'))
    rows = [
        ('studio.launch','application','studio_cli.main','studio','Foreground Streamlit app on 0.0.0.0; restrict network access; no runtime validation performed'),
        ('studio.download','application','studio.download_model','studio','Explicit curated Hub download with visible callback/Stop cleanup; no inference service'),
        ('studio.load','application','studio.load_model','studio','Offline native SmolLM2/CLIP safetensors loading; session-local ownership'),
        ('studio.prepare','application','studio.prepare_model','studio','Reuse complete cache or consent-gated fetch, then local load; staged progress, static review only'),
        ('studio.nla','application','studio_nla.render','studio','Connected CPU-staged Qwen L20 verbalization/reconstruction/edit comparison; no runtime validation'),
        ('studio.text','application','studio.explain_text','studio','Greedy continuation and fixed-target embedding gradient x input; static review only'),
        ('studio.image','application','studio.explain_image','studio','CLIP scoring and crop-aligned 16-step pixel IG, not Grad-ECLIP; static review only'),
        ('studio.unload','application','studio.unload','studio','Release supplied session model; does not delete downloaded cache files'),
        ('nla.load','language','nla.NaturalLanguageAutoencoder.load','nla','Explicit local released Qwen2.5-7B L20 AV/AR safetensors; no automatic downloads; static review only'),
        ('nla.load_av','language','nla.NaturalLanguageAutoencoder.load_av','nla','Release prior stage, memory preflight, load only AV; explicit local assets'),
        ('nla.load_ar','language','nla.NaturalLanguageAutoencoder.load_ar','nla','Release prior stage, memory preflight, load only AR/head; explicit local assets'),
        ('nla.release','language','nla.NaturalLanguageAutoencoder.release','nla','Release model references without deleting checkpoint files'),
        ('nla.encode','language','nla.NaturalLanguageAutoencoder.encode','nla','Activation verbalizer with norm-150 prompt embedding injection and bounded greedy generation; unexecuted'),
        ('nla.decode','language','nla.NaturalLanguageAutoencoder.decode','nla','Released 21-layer activation reconstructor with trained value head; raw vector norms not calibrated'),
        ('nla.reconstruct','language','nla.NaturalLanguageAutoencoder.reconstruct','nla','Activation-to-text-to-activation round trip; normalized MSE/cosine, no empirical quality claim'),
        ('nla.edit','language','nla.NaturalLanguageAutoencoder.edit','nla','Residual-preserving text-decoded activation edit variant; not paper steering reproduction'),
        ('nla.extract','language','nla.extract_nla_activations','','Original Qwen2.5-7B block-20 output, one unpadded source sequence; caller supplies source model'),
        ('nla.patch','intervention','nla.patch_nla_activations','','Exact source token positions and dtype/device, one uncached full forward; caller verifies token identity'),
        ('nla.reconstruction_loss','evaluation','nla.nla_reconstruction_loss','','Differentiable direction-normalized AR MSE component; not an SFT/GRPO trainer'),
        ('clip.similarity','multimodal','clip_tools.clip_similarity','','Differentiable normalized image/text scores; caller-provided projected embeddings'),
        ('clip.scores','multimodal','clip_tools.clip_scores','','Explicit aligned-pair or indexed text targets; no inferred argmax'),
        ('clip.encode_scores','multimodal','clip_tools.encode_clip_scores','','Caller encoders, preprocessing and eval mode; no pretrained adapter'),
        ('clip.grad_eclip','multimodal','clip_tools.grad_eclip','','Source-grounded single-layer single-head intermediate contract; encoder instrumentation required, statically reviewed only'),
        ('clip.token_gradients','multimodal','clip_tools.token_gradient_attribution','','Signed gradient-times-token; separate from Grad-ECLIP'),
        ('clip.patch_map','multimodal','clip_tools.patch_attribution_map','','Explicit rectangular grid and token order'),
        ('clip.component_scores','multimodal','clip_tools.projected_component_scores','','Exact score bookkeeping only for supplied components in final projected space'),
        ('clip.attention_values','multimodal','clip_tools.attention_value_decomposition','','Linear attention head/token decomposition for one query, not full CLIP decomposition'),
        ('multimodal.sparse_concepts','multimodal','multimodal_concepts.sparse_concept_decomposition','','SpLiCE-style nonnegative L1 ISTA variant; explicit calibration and convergence diagnostics'),
        ('multimodal.reconstruct_concepts','multimodal','multimodal_concepts.reconstruct_concepts','','Dictionary-coordinate reconstruction with explicit inverse preprocessing'),
        ('multimodal.intervene_concepts','multimodal','multimodal_concepts.intervene_concepts','','Error-preserving embedding coefficient edits, not causal input edits'),
        ('multimodal.feature_diagnostics','multimodal','multimodal_concepts.modality_feature_diagnostics','','Aligned-feature prevalence/specificity/correlation; not SAE training'),
        ('multimodal.grouped_evidence','multimodal','multimodal_metrics.grouped_evidence_evaluation','','Grouped deletion/insertion and sufficiency/necessity with matched random controls; static review only'),
        ('multimodal.grounding','multimodal','multimodal_metrics.grounded_evidence_metrics','','Weighted spatial/temporal annotation overlap, not faithfulness'),
        ('multimodal.fixed_prefix','multimodal','multimodal_metrics.fixed_prefix_log_probs','','Explicit teacher-forced causal token/sequence scores; multimodal position alignment required'),
        ('multimodal.donor_sweep','multimodal','multimodal_tracing.donor_patching_sweep','','Aligned tensor-site clean/corrupt patching with self-controls and raw recovery'),
        ('multimodal.attention_payload','multimodal','multimodal_tracing.cross_modal_attention_payload','','Bounded descriptive cross-modal attention data; not a visualization UI or causal attribution'),
        ('multimodal.feature_shapley','multimodal','feature_shap.feature_shapley','','Joint feature game with exact or permutation estimator, explicit grouping/replacements; statically reviewed only'),
        ('multimodal.mm_shap','multimodal','feature_shap.mm_shap','','Paper equations 2-3: sum absolute feature Shapley values within modalities, then normalize'),
        ('multimodal.intershap','multimodal','feature_shap.intershap','','Paper equations 5-11: distinct global/local interaction ratios from symmetric half-pair matrices'),
        ('multimodal.score','multimodal','multimodal.MultimodalScorer','','Named tensors to explicit fixed scalar objective; statically reviewed, not runtime tested'),
        ('multimodal.shapley','multimodal','multimodal.modality_shapley','','Exact bounded whole-modality replacement game and half-pair interactions; not feature-level MM-SHAP'),
        ('multimodal.ablation','multimodal','multimodal.modality_ablations','','Leave-one-out and singleton effects under explicit references'),
        ('multimodal.mismatched_pairs','multimodal','multimodal.mismatched_pairs','','Explicit derangement control; descriptive, not causal evidence'),
        ('multimodal.emap','multimodal','multimodal.emap','','Exact two-modality scalar empirical Cartesian additive projection; quadratic pair budget'),
        ('multimodal.reliance','multimodal','multimodal.compare_answer_explanation_reliance','','Compare fixed answer/explanation objectives using absolute modality shares; not a truthfulness metric'),
        ('evaluation.infidelity','evaluation','backends.attribution_infidelity','attribution','Captum perturbation infidelity'),
        ('evaluation.sensitivity','evaluation','backends.attribution_sensitivity','attribution','Captum local max sensitivity'),
        ('concept.cav','representation','concepts.fit_cav','tabular','Linear separator; independent controls required'),
        ('concept.nmf','representation','concepts.nmf_features','tabular','Nonnegative representation decomposition'),
        ('sae.inspect','features','features.SAEAdapter.inspect','sae','OpenAI vendored/SAELens real implementations tested with tiny dictionaries'),
        ('sae.steer','features','features.steer_sae','sae','Error-preserving feature interventions'),
        ('sae.top_examples','features','features.feature_examples','','Feature rankings, no automatic semantic labels'),
        ('sae.local_load','features','features.load_saelens_local','sae','Local config/weight round-trip tested'),
        ('gemma_scope.load','features','features.load_saelens_release','sae','Pretrained loader API; real Google artifacts NOT executed here; opt-in download'),
        ('jlens.local_fit','language','lenses.fit_local_jacobian','','Small exact causal-tail Jacobian estimator'),
        ('jlens.reference_fit','language','lenses.fit_reference_jlens','jlens','Actual Anthropic code tested with tiny GPT-2 and pretrained Gemma4; separate pretrained environment'),
        ('jlens.readout','language','lenses.read_reference_jlens','jlens','J-lens vs logit-lens; pretrained Gemma4 helper also retains held-out and scrambled controls'),
        ('openai.sparse_circuits','language','frontier.inspect_sparse_circuit','openai','Actual OpenAI architecture/hook recorder; random tiny weights'),
        ('goodfire.nano_decomposition','language','frontier.run_goodfire_nano','goodfire-local','Actual vendored nano VPD objective; short CPU training, not converged research'),
        ('goodfire.ember_search','remote','frontier.GoodfireRemote.search','goodfire-remote','Archived hosted SDK; opt-in; live service NOT verified'),
        ('circuits.trace','language','frontier.trace_circuit','tracing','Actual tracing on random tiny model/transcoders; separate environment'),
        ('circuits.edges','language','circuits.top_graph_edges','','Signed direct-edge ranking, not completeness proof'),
        ('diffusion.sample','diffusion','diffusion.sample_diffusers','diffusion','Actual tiny Diffusers image model, matched noise and generators'),
        ('diffusion.intervene','diffusion','diffusion.diffusion_intervention','','Explicit timesteps/branch rows/spatial mask; not full PolypSteer'),
        ('representation.cka','representation','representations.linear_cka','','Matched observations, centered linear CKA'),
        ('representation.probe','representation','representations.ridge_probe','tabular','Train-only standardization'),
        ('concept.sensitivity','representation','representations.concept_sensitivity','','TCAV score primitive, not full TCAV significance study'),
        ('attention.rollout','language','representations.attention_rollout','','Descriptive mean-head rollout, not causal proof'),
        ('language.logit_lens','language','representations.logit_lens','','Caller supplies final normalization/unembedding'),
        ('activation.ablation','intervention','representations.ablate_features','','Explicit neuron/head axis, tensor outputs'),
        ('evaluation.randomization','evaluation','representations.parameter_randomization_check','','Fresh factory model vs trained model explanations'),
        ('tabular.permutation','tabular','tabular_tools.permutation_importance','tabular','Held-out data recommended'),
        ('tabular.pdp_ice','tabular','tabular_tools.partial_dependence_ice','tabular','Brute PDP/ICE, correlated-feature caveats'),
        ('tabular.counterfactual','tabular','tabular_tools.nearest_counterfactual','tabular','Nearest supplied feasible candidate, not causal recourse'),
        ('activation.capture','intervention','activations.capture_activations','','Detached tensor capture'),
        ('activation.patch','intervention','activations.patch_activation','','Masked donor patch'),
        ('activation.steer','intervention','steering.steer','','Addition/positive projection removal'),
        ('concept.direction','intervention','steering.concept_direction','','Matched contrastive means'),
        ('evaluation.perturbation','evaluation','metrics.perturbation_faithfulness','','Deletion/insertion with random controls'),
    ]
    tools.extend(Tool(n,f,'autoexplain.'+e,x,s) for n,f,e,x,s in rows)
    return [t for t in tools if family is None or t.family == family]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family')
    args = parser.parse_args()
    print(json.dumps([asdict(t) for t in list_tools(args.family)], indent=2))


if __name__ == '__main__':
    main()
