# API and limitations

For the optional attribution, SAE, J-lens, circuit, tabular and diffusion APIs added in 0.2, see [Tool integrations](integrations.md). Use `python -m autoexplain.catalog` to list entry points and scope. The original APIs below remain supported.

## `Inspector(model, checkpoint=None)`

Accepts a `torch.nn.Module` and optionally loads its state dict. `Inspector` itself is PyTorch-only. Optional tabular helpers use explicit scikit-learn interfaces in the [tree notebook](notebooks.md).

- `layers()` returns the named-module mapping.
- `suggest()` returns `Recommendation` records containing `method`, `rationale`, `requirements` and `candidate_layers`. Recommendations use structure, not a trial forward pass.
- `input_gradients(inputs, target=None)` returns signed input gradients with the input's shape.
- `gradcam(inputs, layer=None, target=None)` returns normalized `[batch, height, width]` maps. Constant maps normalize to zero.

Targets can be a scalar class index or one index per sample. Both explanation methods temporarily switch modules to evaluation mode and restore individual training flags afterward. They do not populate or clear model parameter gradients. Inputs are detached and cloned. Run outside `torch.inference_mode()`; `torch.no_grad()` is supported through an internal gradient-enabled context.

Grad-CAM requires a spatial tensor-output module that executes once. Reused modules, compiled graphs, distributed wrappers, quantized models and mixed-precision edge cases are not validated. Batch-coupled models need special treatment because the implementation differentiates the sum of selected sample scores.

## `concept_direction(positive, negative)`

Computes a unit vector from the difference of activation means. Both inputs must be nonempty `[observations, features]` tensors with the same feature width and compatible devices. Zero and nonfinite directions raise `ValueError`. Input activations are detached; direction estimation is not differentiable.

## `steer(model, layer, direction, *, strength=1.0, feature_axis=-1, mode="add")`

A context manager that registers one forward hook and removes it on exit. Supports `add` and `remove_positive_projection`. Directions are normalized internally and moved to the activation device/dtype when applied. Direction width must match the selected feature axis. Outputs must be floating tensors; tuple/dictionary outputs need explicit adapters.

The hook changes all selected activations during the context, including repeated module calls. It does not distinguish diffusion timesteps, attention branches or token roles. Do not share a hooked model between concurrent requests. Very large strengths or low-precision overflow can produce invalid outputs; callers must check finiteness and task behavior.

## Additional functional APIs

Top-level imports include `integrated_gradients`, `smoothgrad`, `occlusion`, `layercam`, `capture_activations`, `patch_activation` and `perturbation_faithfulness`. See [Methods](methods.md) for exact signatures, result dataclasses, shapes and citations. The existing `Inspector` API remains unchanged. Tutorial-specific adapters and compact models live in `autoexplain.adapters`, `autoexplain.toys` and `autoexplain.tabular`.

## Scientific limitations

Explanations are method-dependent views of model behavior. Validate them with parameter randomization, perturbation controls, alternative target choices and repeated inputs. Deletion/insertion curves with matched random rankings are implemented in `perturbation_faithfulness`; broader parameter-randomization studies remain caller responsibilities. Existing unit tests establish implementation properties, not explanation faithfulness or clinical validity.
