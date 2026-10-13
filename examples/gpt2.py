"""GPT-2 demonstration using the AutoExplain persistent XAI harness."""

from autoexplain import Harness


def main():
    h = Harness()
    info = h.open_model("gpt2")
    print(f"Loaded {info['model_name']} with {info['n_layers']} layers and {info['n_heads']} heads.")

    arch = h.inspect()
    print(f"Architecture: d_model={arch['d_model']}, d_head={arch['d_head']}, params={arch['n_params']:,}")

    prompt = "Alice gave Bob a book."
    baseline = h.run(prompt)
    print(f"\nBaseline top prediction: {baseline.top_tokens[0]['token']!r} (prob={baseline.top_tokens[0]['probability']:.4f})")

    obs = h.observe(layer=5)
    print(f"Layer 5 state: residual_norm={obs['residual_norm']:.2f}, head_norms={len(obs['head_norms'])} heads measured")

    # Ablate attention head 3 at layer 5
    h.intervene(layer=5, head=3, method="zero")
    modified = h.run(prompt)
    print(f"Modified top prediction: {modified.top_tokens[0]['token']!r} (prob={modified.top_tokens[0]['probability']:.4f})")

    # Clear interventions without reloading model weights
    h.reset()

    result = h.evaluate(baseline, modified)
    print("\nEvaluation:")
    print(f"  Max logit diff:      {result['max_logit_diff']:.4f}")
    print(f"  Mean logit diff:     {result['mean_logit_diff']:.4f}")
    print(f"  Cosine similarity:   {result['cosine_similarity']:.6f}")
    print(f"  KL divergence:       {result['kl_divergence']:.4f}")
    print(f"  Prediction changed:  {result['top_prediction_changed']}")

    saved_path = h.record("experiment_001")
    print(f"\nSaved experiment record to: {saved_path}")


if __name__ == "__main__":
    main()
