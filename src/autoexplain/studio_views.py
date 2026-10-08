"""Presentation and bounded, allowlisted exports for the local studio."""
import copy
import html
import json
import re

import streamlit as st


def export_record(record):
    """Keep explanation inputs, never arbitrary checkpoint/processor metadata."""
    keys = ("method", "score", "target_id", "target_token", "results", "settings",
            "target_index", "candidates", "logits", "candidate_softmax", "input_sha256",
            "completeness_residual", "signed_channel_sum", "mean_absolute_channel_attribution")
    clean = {key: copy.deepcopy(record[key]) for key in keys if key in record}
    metadata = record.get("model", {})
    from autoexplain.studio import LIBRARY
    model_id = metadata.get("model_id")
    clean["model"] = {"model_id": model_id if model_id in LIBRARY else "local model"}
    for key, choices in (("device", ("cpu", "cuda:0")),
                         ("dtype", ("float32", "float16", "bfloat16")),
                         ("kind", ("text", "image"))):
        if metadata.get(key) in choices:
            clean["model"][key] = metadata[key]
    revision = str(metadata.get("resolved_revision", ""))
    clean["model"]["resolved_revision"] = revision if re.fullmatch(r"[0-9a-f]{40}", revision) else "unverified local snapshot"
    # The supported CLIP crop is 224×224; do not retain unbounded image arrays.
    for key in ("signed_channel_sum", "mean_absolute_channel_attribution"):
        if key in clean:
            clean[key] = [row[:224] for row in clean[key][:224]]
    return clean


def token_highlights(row):
    values = row["attribution"]
    scale = max((abs(value) for value in values), default=0) or 1
    pieces = []
    for token, value in zip(row.get("readable_tokens", row["tokens"]), values):
        readable = token if "readable_tokens" in row else token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
        color = "25,110,190" if value >= 0 else "195,65,35"
        alpha = 0.10 + 0.55 * abs(value) / scale
        pieces.append(f'<span style="background:rgba({color},{alpha:.3f});border-radius:3px" '
                      f'title="signed attribution: {value:.6g}">{html.escape(readable)}</span>')
    st.markdown('<div style="white-space:pre-wrap;line-height:2;font-family:monospace">'
                + "".join(pieces) + '</div>', unsafe_allow_html=True)
    st.caption("Blue: positive contribution · Orange: negative contribution · Intensity: relative absolute magnitude within this prompt. Token boundaries may split words.")


def text_result(record):
    st.write(f"Fixed next-token target: {record['target_token']!r} (ID {record['target_id']})")
    rows = record["results"]
    for column, row, label in zip(st.columns(len(rows)), rows, ("Original", "Edited prompt")):
        with column, st.container(border=True):
            st.markdown(f"**{label}**")
            st.text(row["prompt"])
            st.markdown("**Generated continuation**")
            st.text(row["continuation"])
            st.write({"target_logit": row["target_logit"], "target_probability": row["target_probability"]})
            token_highlights(row)
            with st.expander("Numeric token inspector"):
                st.dataframe({"token": row["tokens"], "token ID": row["input_ids"],
                              "signed gradient × input": row["attribution"]}, hide_index=True)
    if len(rows) == 2:
        st.metric("Edited − original fixed-target logit", f"{rows[1]['target_logit'] - rows[0]['target_logit']:.6g}")
        st.caption("Both prompts explain the same token. Prompt editing is an input intervention, not an activation patch; token positions may change.")
    st.caption("Attribution explains one next-token logit, NOT the whole continuation. Gradient × input is local sensitivity, not a causal proof.")


def image_result(saved):
    import matplotlib.pyplot as plt
    record = saved["record"]
    target = record["target_index"]
    st.write(f"Fixed candidate {target}: {record['candidates'][target]}")
    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)
    colormap = st.selectbox("Overlay colormap", ["inferno", "viridis", "magma", "cividis"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    try:
        for axis in axes:
            axis.imshow(saved["display"])
            axis.axis("off")
        axes[0].set_title("Model crop")
        overlay = axes[1].imshow(saved["heatmap"], cmap=colormap, alpha=opacity, vmin=0, vmax=1)
        axes[1].set_title("IG magnitude")
        fig.colorbar(overlay, ax=axes[1], fraction=0.046, label="Relative magnitude")
        st.pyplot(fig)
    finally:
        plt.close(fig)
    with st.expander("Candidate scores and numeric inspector"):
        st.dataframe({"candidate": record["candidates"], "CLIP logit": record["logits"],
                      "candidate-set softmax": record["candidate_softmax"]}, hide_index=True)
        st.write({"signed_IG_completeness_residual": record["completeness_residual"]})
    st.caption("Overlay: mean absolute channel attribution normalized per image. Signed channel sums are exported (at most 224×224). Input-pixel IG, not Grad-ECLIP. Baseline: zero normalized pixels (channel-mean RGB). Candidate softmax is relative to supplied texts, not calibrated confidence. A small residual is not semantic validation.")


def results(saved):
    record = saved["record"]
    st.caption(record["method"] + "; score: " + record["score"])
    if "results" in record:
        text_result(record)
    else:
        image_result(saved)
    st.download_button("Export this explanation JSON", json.dumps(export_record(record), indent=2, allow_nan=False),
                       "autoexplain-result.json", "application/json")
    st.caption("Exports include your input text and explanation data; review sensitive inputs before sharing. Authentication data, local checkpoint paths, arbitrary revisions, processor metadata and original image bytes are not collected for export.")
    from .studio_storage import save_control
    save_control("text" if "results" in record else "image", export_record(record),
                 key="studio_result_local", default_name="My explanation")


def history():
    with st.expander("Session history · compare runs and export", expanded=False):
        runs = st.session_state.get("studio_history", [])
        st.caption("This browser session only, latest 8 runs. No files are saved automatically. Model unload preserves history; restart loses it. Exports contain supplied text.")
        if not runs:
            st.info("Run an explanation to start a named comparison.")
            return
        options = list(range(len(runs)))
        selected = st.multiselect("Choose up to two named runs", options, default=options[-2:],
                                  max_selections=2, format_func=lambda i: f"{i + 1} · {runs[i]['name']}")
        for column, index in zip(st.columns(max(1, len(selected))), selected):
            with column:
                run = runs[index]
                record = run["record"]
                st.write(run["name"])
                st.caption(record["model"]["model_id"])
                if "results" in record:
                    st.write(f"Fixed target: {record['target_token']!r} / {record['target_id']}")
                    for row in record["results"]:
                        st.text(row["prompt"])
                        st.text(row["continuation"])
                        st.write({"target_logit": row["target_logit"]})
                else:
                    st.write({"target": record["candidates"][record["target_index"]],
                              "logit": record["logits"][record["target_index"]]})
        st.caption("Separate runs may use different targets or models: compare their metadata before interpreting score changes. For a controlled text edit, use the paired-prompt workflow above.")
        st.download_button("Export named comparison JSON", json.dumps([runs[i] for i in selected], indent=2, allow_nan=False),
                           "autoexplain-comparison.json", "application/json", disabled=not selected)
        if st.button("Clear session history"):
            st.session_state.pop("studio_history", None)
            st.rerun()
