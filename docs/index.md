# AutoExplain

Explain a model's prediction, intervene on its internal features, and compare the result in a runnable experiment.

AutoExplain 0.2 combines native methods with Captum, CAM, SHAP, sparse features, J-lens, circuit tracing, Goodfire nano decomposition and Diffusers. See [tool integrations](integrations.md) for installation and exact support boundaries. The project aims to cover more model families, but automatic support for any AI model is not available yet.

## Start with an experiment

1. Open the [playground guide](playground.md) and run the local visual app.
2. Change the target class and inspect the resulting heatmap.
3. Change steering strength and compare the prediction against the unchanged baseline.
4. Compare perturbation curves against random rankings and try donor-feature patching.
5. Run the [thirteen notebooks](notebooks.md), then follow [Your model](models.md) for your own architecture.

The [method guide](methods.md) documents formulas, citations and limitations. [Validation evidence](validation.md) records local checks; none establishes state-of-the-art explanation performance.

The bundled model classifies synthetic vertical and horizontal bars. It trains on CPU without downloading datasets or weights. That makes the mechanics reproducible; it does not demonstrate performance on natural images or medical tasks.

## Choose methods by capability

A CNN can expose spatial activations for Grad-CAM. A transformer or diffusion component may expose token features for steering. Those capabilities do not establish which layer, score, concept direction, or control experiment is appropriate. AutoExplain returns candidate methods with their requirements and keeps unsupported tasks explicit.

Read the [diffusion guide](diffusion.md) for the distinction between generic steering primitives and PolypSteer reproduction, or the [API reference](api.md) for exact input contracts.
