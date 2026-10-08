# AutoExplain

Explain a model's prediction, intervene on its internal features, and compare the result in a runnable experiment.

AutoExplain 0.2 combines native methods with Captum, CAM, SHAP, sparse features, J-lens, circuit tracing, Goodfire nano decomposition and Diffusers. See [tool integrations](integrations.md) for installation and exact support boundaries. The project aims to cover more model families, but automatic support for any AI model is not available yet.

## Start with an experiment

1. Follow the [local app guide](playground.md), install `[studio]`, and launch `autoexplain-studio`.
2. Start in **Recorded pipelines**: inspect actual Gemma 4 inputs, next-token predictions, route-aware attributions and before/after donor patches. Reports are from prior real runs, not dummy data or fresh inference.
3. Use **Live cache-only rerun** to explicitly repeat one fixed Gemma pipeline with a compatible backend, or choose **Text explanation** / **Image explanation** for new inputs and named comparisons.
4. Choose **Activation language** for staged Qwen NLA verbalization, reconstruction and description-edit controls. **Synthetic CNN tutorial** remains a secondary teaching example, not the default pipeline.
5. Run the [thirteen CPU notebooks](notebooks.md) and [two real pretrained Gemma 4 examples](pretrained.md), then follow [Your model](models.md) for your own architecture.

The [method guide](methods.md) documents formulas, citations and limitations. [Validation evidence](validation.md) records local checks; none establishes state-of-the-art explanation performance.

The optional synthetic page trains a bar classifier on CPU without model downloads. Pretrained pages connect SmolLM2-135M token attribution, CLIP image-pixel integrated gradients and the released Qwen2.5-7B block-20 [NLA pair](nla.md). NLA loads one model at a time on CPU with memory/disk guidance; it remains a large, potentially slow workflow. These pages have received **static review only, not runtime validation**. See the [UX reference notes](studio-ux.md) for the design basis.

## Choose methods by capability

A CNN can expose spatial activations for Grad-CAM. A transformer or diffusion component may expose token features for steering. Those capabilities do not establish which layer, score, concept direction, or control experiment is appropriate. AutoExplain returns candidate methods with their requirements and keeps unsupported tasks explicit.

Read the [diffusion guide](diffusion.md) for the distinction between generic steering primitives and PolypSteer reproduction, or the [API reference](api.md) for exact input contracts.
