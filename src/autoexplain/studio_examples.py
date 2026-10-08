"""Recorded real Gemma pipelines and an explicitly requested offline rerun.

The recorded view uses only JSON resources: it never imports Transformers or
loads a model. The isolated runner owns the separate Transformers-5 environment.
"""
import hashlib
from importlib import resources
import json

from . import studio_example_runner as runner
from . import studio_jobs


def _saved(name):
    raw = resources.files("autoexplain").joinpath("_examples", name).read_bytes()
    return json.loads(raw), raw, hashlib.sha256(raw).hexdigest()


def _table(st, rows):
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _provenance(st, report, name, raw, digest):
    with st.expander("Source, environment and exact numerical report"):
        st.caption(f"Recorded source: examples/outputs/{name} · bundled byte-for-byte; independent of working directory")
        st.code(f"SHA256 {digest}\nModel revision {report['model']['revision']}", language=None)
        metadata = {key: value for key, value in report.items()
                    if key in ("model", "gpu_name", "gpu_peak_allocated_gib", "gpu_peak_reserved_gib",
                               "cpu_peak_rss_gib", "python_version", "torch_version", "transformers_version",
                               "threads", "total_seconds", "experiment_seconds", "fit_seconds")}
        st.json(metadata)
        if "transformers_version" not in report:
            st.caption("This J-lens JSON does not record hardware or dependency versions; do not infer them from the other run. docs/pretrained.md describes the measured environment separately.")
        st.download_button("Download exact recorded JSON", raw, file_name=name,
                           mime="application/json", key="download_" + name)
        st.json(report, expanded=False)


def _pipeline(st, report, *, key, allow_save=True):
    pairs = report["pairs"]
    selected = report.get("selection", {}).get("pair", 0)
    selected = selected if type(selected) is int and selected in range(len(pairs)) else 0
    index = st.selectbox("Fixed donor → recipient pair", range(len(pairs)), index=selected,
                         format_func=lambda i: pairs[i]["clean_prompt"] + " → " + pairs[i]["recipient_prompt"],
                         key=key + "_pair")
    pair = pairs[index]
    patches = pair["patches"]
    default = next((i for i, row in enumerate(patches)
                    if row["layer"] == 8 and row["position_name"] == "changed_token"), 0)
    selected_patch = report.get("selection", {}).get("patch", default)
    default = selected_patch if type(selected_patch) is int and selected_patch in range(len(patches)) else default
    patch_index = st.selectbox("Residual intervention", range(len(patches)), index=default,
                               format_func=lambda i: f"Layer {patches[i]['layer']} · {patches[i]['position_name']} · position {patches[i]['position']}",
                               key=key + "_patch")
    patch = patches[patch_index]
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("1 · Input")
        st.caption("Donor prompt — exact cloze input, not a chat instruction")
        st.code(pair["clean_prompt"], language=None)
        st.caption("Recipient prompt — kept fixed during patching")
        st.code(pair["recipient_prompt"], language=None)
        st.write(f"Fixed score: logit({pair['target']!r}) − logit({pair['foil']!r})")
        st.caption("Single-token targets include a leading space. BOS is included; positions and layers are zero-indexed.")
        with st.expander("Exact donor tokens & positions"):
            _table(st, [{"position": i, "donor token": token} for i, token in enumerate(pair["token_labels"])])
    with right.container(border=True):
        st.subheader("2 · Next-token output")
        st.write("Donor top-1:", repr(pair["clean_top5"][0]["text"]))
        st.write("Recipient top-1:", repr(pair["recipient_top5"][0]["text"]))
        st.caption("Softmax probabilities below refer to the immediate next token, not a generated answer or factual confidence.")
        with st.expander("Exact next-token probabilities"):
            _table(st, [{"input": label, "token": repr(row["text"]), "token ID": row["id"], "probability": row["probability"]}
                        for label, field in (("donor", "clean_top5"), ("recipient", "recipient_top5")) for row in pair[field]])
    with st.container(border=True):
        st.subheader("3 · Compare the intervention")
        st.write(f"Copy donor post-block residual at layer {patch['layer']}, {patch['position_name']} position {patch['position']} into the recipient.")
        before, after, donor = st.columns(3)
        before.metric("Before · recipient top token", repr(pair["recipient_top5"][0]["text"]))
        before.metric("Before · fixed-target margin", pair["recipient_margin"])
        after.metric("After · patched top token", repr(patch["top_token"]))
        after.metric("After · fixed-target margin", patch["margin"], delta=patch["margin_change"])
        donor.metric("Donor · fixed-target margin", pair["clean_margin"])
        donor.metric("Margin recovery ratio", patch["recovery_ratio"])
        st.caption("Same fixed target − foil logit score before and after. Margins/recovery are NOT probabilities or success rates; a positive margin can coexist with an unrelated top token. Patched-token probabilities were not recorded.")
        if patch["top_token_id"] != pair["target_id"]:
            st.warning(f"Retained failure: desired {pair['target']!r} is NOT top-1; the actual next token is {patch['top_token']!r}.")
        if patch["trivial_final_readout_control"]:
            st.warning("Trivial positive control: final-layer last-token replacement copies the donor answer readout. This is not circuit discovery.")
        st.caption("Token IDs, per-layer token embeddings and shared-KV routes otherwise remain unchanged. This is a local residual intervention, not complete concept transfer.")
        with st.expander("Random & self-patch controls · exact scores"):
            _table(st, [{"seed": seed, "norm-matched random margin": value}
                        for seed, value in zip(report["random_seeds"], patch["random_margins"])])
            _table(st, pair["self_controls"])
            st.caption("Self-patches should preserve logits. Random directions match the donor-minus-recipient norm; three seeds are controls, not a significance test.")
    with st.expander("Signed route-aware attribution", expanded=True):
        st.caption("Donor prompt · gradient × activation of the fixed target − foil margin. Main and per-layer embedding routes are separate conditional sensitivities, not additive completeness-certified explanations. Negative values are retained.")
        from .studio_views import token_highlights
        rows = []
        for route, scores in pair.get("embedding_attribution", {}).items():
            st.write(route)
            token_highlights({"tokens": pair["token_labels"],
                              "attribution": scores["gradient_x_activation"]})
            for i, token in enumerate(pair["token_labels"]):
                rows.append({"route": route, "position": i, "token": token,
                             "signed gradient × activation": scores["gradient_x_activation"][i],
                             "gradient L2": scores["gradient_l2"][i]})
        with st.expander("Exact signed attribution values"):
            _table(st, rows)
    with st.expander("All interventions and plain logit-lens controls"):
        _table(st, patches)
        _table(st, pair["readouts"])
        st.write("Final-readout margin error:", pair.get("final_readout_margin_error", "Not evaluated at this layer"))
        st.caption("Plain logit lens includes the final norm, head and softcap. Layer-34 last-token donor replacement is a trivial readout control.")
    with st.expander("Study limitations"):
        for limitation in report["limitations"]:
            st.write("• " + limitation)
    if allow_save:
        from .studio_storage import save_control
        exported = {field: report[field] for field in ("layers", "random_seeds", "pairs", "status", "limitations",
                    "total_seconds", "experiment_seconds", "python_version", "torch_version", "transformers_version", "threads") if field in report}
        exported["model"] = {field: report["model"][field] for field in ("model_id", "revision", "dtype", "quantized", "decoder_device", "per_layer_embedding_device") if field in report["model"]}
        exported["origin"] = report.get("origin", "recorded real Gemma study; not a new run")
        exported["selection"] = {"pair": index, "patch": patch_index}
        save_control("gemma", exported, key=key + "_local", default_name="Gemma intervention")


def _jlens(st):
    report, raw, digest = _saved("gemma4_jlens.json")
    st.subheader("Recorded real J-lens · layer 8")
    st.caption("Saved fitted-lens diagnostic, not a live J-lens reproduction. Four fit prompts and six held-out capital prompts; no pretrained SAE or discovered circuit is claimed.")
    with st.container(border=True):
        st.write("Fit inputs")
        for prompt in report["fit_prompts"]:
            st.code(prompt, language=None)
        st.write("Estimator settings", report["estimator"])
    rows = report["readouts"]
    index = st.selectbox("Held-out readout input", range(len(rows)), format_func=lambda i: rows[i]["prompt"], key="saved_jlens_prompt")
    row = rows[index]
    st.code(row["prompt"], language=None)
    st.write("Actual next-token logits (not lens predictions)")
    _table(st, row["actual_next_token_top5"])
    for field, label in (("country_readouts", "Country token"), ("last_readouts", "Last token")):
        values = row[field]
        st.write(f"{label} · position {row['positions']['country' if field == 'country_readouts' else 'last']}")
        columns = st.columns(3)
        for column, method in zip(columns, ("jacobian_lens", "logit_lens", "scrambled_lens")):
            with column.container(border=True):
                st.caption(method)
                _table(st, values[method])
    st.write("Country-token ranks", row["country_token_ranks"])
    st.caption(report["rank_definition"] + f" · scrambled-matrix seed {report['scramble_seed']}")
    st.caption("Fitted-lens scores are readout logits, not probabilities. Country readability is not next-token correctness. The France last-token fitted top five do not contain Paris.")
    for limitation in report["limitations"]:
        st.write("• " + limitation)
    _provenance(st, report, "gemma4_jlens.json", raw, digest)


def _live(st):
    st.subheader("Edit inputs → run → compare")
    st.caption("LIVE · Gemma 4 · pinned local weights · nothing runs until you click")
    defaults = ("The capital of France is", "The capital of Germany is", " Paris", " Berlin")
    left, right = st.columns(2)
    with left:
        donor = st.text_area("Donor prompt", value=defaults[0], max_chars=2000, key="gemma_donor")
        target = st.text_input("Fixed target token", value=defaults[2], max_chars=100, key="gemma_target")
    with right:
        recipient = st.text_area("Recipient prompt", value=defaults[1], max_chars=2000, key="gemma_recipient")
        foil = st.text_input("Fixed foil token", value=defaults[3], max_chars=100, key="gemma_foil")
    inputs = [donor, recipient, target, foil]
    layer = st.selectbox("Intervention layer", runner.LAYERS, index=1, key="live_gemma_layer")
    st.caption("2–32 native tokens per prompt; equal length; exactly one changed token before the final token. Target and foil must be distinct single tokens (often with leading spaces). Invalid inputs stop before weight loading.")
    with st.expander("Backend & execution budget"):
        st.code(runner.interpreter(), language=None)
        st.caption("Dedicated Gemma backend; separate from the Studio/NLA environment. One pair, one layer, both patch positions, three random seeds, self-controls and signed gradients; 180 seconds / 1 MiB output. ≥18 GiB available RAM and ≥6 GiB free GPU memory, plus decoder headroom. No downloads, installs, quantization or fallback.")
    status = st.empty()
    def progress(elapsed, message):
        studio_jobs.stage("gemma", message)
        status.info(f"{elapsed}s · {message}")
    if st.button("Check backend, pinned cache and resources", key="gemma_check"):
        st.session_state.pop("gemma_readiness", None)
        try:
            with studio_jobs.task("gemma", "Gemma readiness"):
                st.session_state.gemma_readiness = runner.execute(progress=progress)
            if not st.session_state.gemma_readiness.get("ready"):
                studio_jobs.failure("gemma", "Not ready: open readiness diagnostics; complete local prerequisites or use recorded evidence. No model loaded.")
            status.empty()
        except Exception as exc:
            message = str(exc) if isinstance(exc, RuntimeError) else "Backend check failed; no model loaded."
            studio_jobs.failure("gemma", message)
            status.error(message)
    if "gemma_readiness" in st.session_state:
        ready = st.session_state.gemma_readiness
        st.write("**Ready · cached model and resources available**" if ready.get("ready") else "**Not ready · check local prerequisites**")
        with st.expander("Readiness diagnostics"):
            st.json(ready)
        st.caption("Last check only; Run rechecks. Cache presence does not authenticate weights.")
    else:
        st.caption("Cache / dependencies / resources: not checked in this session.")
    busy = st.session_state.get("studio_model") is not None
    if busy:
        st.warning("Unload this session's Studio model before running Gemma. Concurrent example reruns are also rejected by an interprocess lock.")
    consent = st.checkbox("Run the real cached model now (no downloads)", key="gemma_run_consent")
    st.caption("Cancel with top-right Stop. The owned subprocess is terminated and reaped before the task becomes Cancelled. Last completed output and draft inputs stay available; nothing retries automatically.")
    if st.button("Run bounded real pipeline", type="primary", disabled=busy or not consent or any(not x.strip() for x in inputs), key="gemma_run"):
        try:
            with studio_jobs.task("gemma", "Gemma live run"):
                report = runner.execute(run=True, inputs=inputs, layer=layer, progress=progress)
            st.session_state.gemma_live_result = report
            status.success("Live cache-only run completed; results below are separate from saved evidence.")
        except Exception as exc:
            message = str(exc) if isinstance(exc, RuntimeError) else "Live run failed. Check inputs and local readiness."
            studio_jobs.failure("gemma", message)
            status.error(message)
    studio_jobs.render("gemma")
    report = st.session_state.get("gemma_live_result")
    if report is not None:
        st.subheader("Live result · last completed run (not the current selector settings)")
        _pipeline(st, report, key="live_result")
        st.download_button("Download live report", json.dumps(report, indent=2, allow_nan=False),
                           file_name="gemma4_live.json", mime="application/json")
        with st.expander("Live metadata and full report"):
            st.json(report)


def render():
    """Studio entry point; safe to open without a Gemma backend or cached weights."""
    import streamlit as st
    st.title("Explore a real explanation")
    st.caption("RECORDED · Gemma 4 · no inference on opening · next-token scope")
    report, raw, digest = _saved("gemma4_pretrained.json")
    summary, action = st.columns([3, 1])
    summary.write("Compare the recipient’s next token **before → after** one residual intervention. Inspect recorded evidence, or edit the selected example.")
    if action.button("Use this example", type="primary", use_container_width=True):
        pair = report["pairs"][st.session_state.get("recorded_pair", 0)]
        for key, field in (("gemma_donor", "clean_prompt"), ("gemma_recipient", "recipient_prompt"), ("gemma_target", "target"), ("gemma_foil", "foil")):
            st.session_state[key] = pair[field]
        st.session_state.studio_example_view = "Edit & run"
        st.rerun()
    view = st.segmented_control("Pipeline", ["Recorded intervention", "Recorded J-lens", "Edit & run"],
                                default="Recorded intervention", key="studio_example_view")
    if view == "Edit & run":
        _live(st)
    elif view == "Recorded J-lens":
        _jlens(st)
    else:
        _pipeline(st, report, key="recorded")
        _provenance(st, report, "gemma4_pretrained.json", raw, digest)
