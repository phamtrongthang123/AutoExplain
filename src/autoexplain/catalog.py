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
        ('jlens.reference_fit','language','lenses.fit_reference_jlens','jlens','Actual Anthropic code tested with small Hugging Face GPT-2'),
        ('jlens.readout','language','lenses.read_reference_jlens','jlens','J-lens vs logit-lens, random model mechanics only'),
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
