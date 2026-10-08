"""Question-first, session-local AutoExplain Streamlit workspace."""
import hashlib
from importlib.metadata import PackageNotFoundError, version

import streamlit as st

from autoexplain import studio_theme, studio_views, studio_jobs


def _unload():
    from autoexplain import studio

    studio.unload(st.session_state.pop("studio_model", None))
    st.session_state.pop("studio_loaded_settings", None)


def _progress(task_key):
    status = st.empty()
    bar = st.progress(0.0)

    def update(stage, current, total):
        # Streamlit UI calls are native Stop interruption boundaries. Never catch
        # BaseException here: cancellation must unwind backend cleanup blocks.
        studio_jobs.stage(task_key, stage)
        status.info(str(stage))
        bar.progress(min(1.0, max(0.0, current / total)) if total > 0 else 0.0)
    return update


def _library(kind):
    from autoexplain import studio

    st.subheader("Prepare your model")
    model_id = st.selectbox("Recommended model", [key for key, spec in studio.LIBRARY.items() if spec["kind"] == kind],
                            format_func=lambda key: key.split("/")[-1])
    spec = studio.LIBRARY[model_id]
    st.caption(spec["description"])
    try:
        recommendation = studio.recommend(model_id)
    except Exception as exc:
        st.warning(studio.safe_error(exc))
        recommendation = {"device": "cpu", "dtype": "float32", "guidance": "Live fit information unavailable; CPU float32 is the conservative fallback.", "resources": {}}
    st.caption(f"Recommended: {'GPU' if recommendation['device'].startswith('cuda') else 'CPU'} · cache ≈ {spec['disk_gb']} GB · {spec['runtime_gb']}")
    with st.expander("Diagnostics & local snapshot", expanded=False):
        st.caption(recommendation["guidance"])
        st.json(recommendation["resources"])
        st.caption("Disk estimate refers to the home filesystem, not necessarily a relocated model cache.")
        revision = st.text_input("Hub revision", "main", max_chars=128, key=f"studio_revision_{kind}")
        path = st.text_input("Trusted local safetensors snapshot (optional)", max_chars=2048, key=f"studio_path_{kind}")
        device = st.selectbox("Device", ["cpu", "cuda:0"], index=1 if recommendation["device"] == "cuda:0" else 0, key=f"studio_device_{kind}")
        dtypes = ["float32"] if device == "cpu" else ["float32", "float16", "bfloat16"]
        preferred = recommendation["dtype"]
        dtype = st.selectbox("Dtype", dtypes, index=dtypes.index(preferred) if preferred in dtypes else 0)
        st.caption("Local snapshots must match the selected architecture and include tokenizer/processor assets. No pickle weights or remote Python code. Architecture checks do not authenticate local provenance.")
    settings = {"model_id": model_id, "revision": revision, "path": path, "device": device, "dtype": dtype}
    try:
        ready = studio.readiness(model_id, revision=revision, path=path)
    except Exception as exc:
        ready = {"downloaded": False, "message": studio.safe_error(exc)}
    loaded = st.session_state.get("studio_model")
    matches = loaded is not None and settings == st.session_state.get("studio_loaded_settings")
    st.write("**Ready to explain**" if matches else "**On device · prepare to load**" if ready["downloaded"] else "**Not on device · consent required**")
    st.caption(ready["message"])
    if loaded is not None and not matches:
        st.warning("A different model/settings selection is loaded. Prepare replaces it; explanation is disabled until the selection is ready.")
    st.link_button("Model card and license", "https://huggingface.co/" + model_id)
    consent = st.checkbox("Allow fetching this selected model from Hugging Face under its license if files are missing.", key=f"studio_consent_{model_id}_{revision}_{bool(path)}")
    st.caption("Prepare reuses local files. Missing files download only with consent; inference stays offline.")
    prepare, unload = st.columns(2)
    if prepare.button("Prepare model", type="primary", disabled=not ready["downloaded"] and (not consent or bool(path.strip()))):
        _unload()
        try:
            with studio_jobs.task("prepare", "Prepare model"):
                callback = _progress("prepare")
                loaded = studio.prepare_model(**settings, consent=consent, progress=callback)
            st.session_state.studio_model = loaded
            st.session_state.studio_loaded_settings = settings
            st.rerun()
        except Exception as exc:
            studio_jobs.failure("prepare", studio.safe_error(exc))
            st.error(studio.safe_error(exc))
    # Preparation releases the previous object before loading its replacement.
    # On failure, never return that now-unloaded local reference to the input tab.
    loaded = st.session_state.get("studio_model")
    matches = loaded is not None and settings == st.session_state.get("studio_loaded_settings")
    if unload.button("Unload session model", disabled=loaded is None):
        _unload()
        st.rerun()
    with st.expander("Memory, cancellation & access"):
        st.caption("Unload releases only this session's model, never the cache. Use top-right Stop; a load, forward or backward may finish before cancellation. Fit is an estimate, not a guarantee. For gated files, authenticate with the Hub CLI outside this app.")
    studio_jobs.render("prepare")
    return loaded if matches else None


def _save(identity, record, name, **display):
    record = studio_views.export_record(record)
    saved = {"identity": identity, "record": record, **display}
    st.session_state.studio_result = saved
    history = st.session_state.get("studio_history", [])
    # Store no original image bytes, processor metadata or loaded objects.
    st.session_state.studio_history = (history + [{"name": name.strip()[:80] or "Untitled explanation", "record": record}])[-8:]


def _compose(loaded, kind):
    from autoexplain import studio

    st.subheader("Input")
    name = st.text_input("Run / comparison name", "My explanation", max_chars=80, key="studio_run_name")
    if kind == "text":
        example = st.selectbox("Try a question", ["Which words support a capital prediction?", "Does changing sentiment change the next token?", "Write my own"], key="studio_question")
        defaults = {"Which words support a capital prediction?": ("The capital of France is", "The capital of Italy is"),
                    "Does changing sentiment change the next token?": ("The film was wonderful and I felt", "The film was terrible and I felt"),
                    "Write my own": ("", "")}
        prompt = st.text_area("Original prompt", defaults[example][0], max_chars=8000, key=f"studio_prompt_{example}")
        comparison = st.text_area("Edited prompt (optional, same target)", defaults[example][1], max_chars=8000, key=f"studio_edit_{example}")
        choice = st.radio("Output to explain", ["Original prompt's top next token", "Choose one fixed token"], horizontal=True, key="studio_target_choice")
        target = st.text_input("Fixed target text (must tokenize to exactly one token)", max_chars=100, key="studio_target_text") if choice == "Choose one fixed token" else ""
        with st.expander("Generation budget", expanded=False):
            budget = st.slider("Continuation tokens", 1, 64, 32, key="studio_budget")
        st.caption("At most 256 input tokens per prompt. Greedy base-model continuation, no chat template. The original target stays fixed for the edited prompt.")
        identity = studio.fingerprint([loaded.metadata if loaded else None, prompt, comparison, choice, target, budget])
        if st.button("Explain and compare text", type="primary", disabled=loaded is None or not prompt.strip() or (choice == "Choose one fixed token" and not target)):
            try:
                with studio_jobs.task("text", "Explain text"):
                    record = studio.explain_text(loaded, prompt, comparison=comparison, target_text=target,
                                                 max_new_tokens=budget, progress=_progress("text"))
                # Native decoding makes common whitespace tokens legible. Raw IDs
                # and tokenizer pieces remain available in the numeric inspector.
                for row in record["results"]:
                    row["readable_tokens"] = [loaded.processor.decode([token]) for token in row["input_ids"]]
                _save(identity, record, name)
            except Exception as exc:
                studio_jobs.failure("text", studio.safe_error(exc))
                st.error(studio.safe_error(exc))
    else:
        st.caption("Example question: which pixels favor ‘a photograph of a cat’ over ‘a photograph of a dog’?")
        upload = st.file_uploader("Image · PNG/JPEG/WebP · ≤10 MiB, ≤4096 per axis, ≤16 million pixels", type=["png", "jpg", "jpeg", "webp"])
        texts = st.text_area("Candidate texts, one per line (2–16; ≤77 tokens each)", "a photograph of a cat\na photograph of a dog", max_chars=8000, key="studio_candidates")
        candidates = [line.strip() for line in texts.splitlines() if line.strip()]
        target = st.selectbox("Fixed candidate to explain", list(range(len(candidates))), format_func=lambda i: candidates[i]) if candidates else 0
        content = upload.getvalue() if upload is not None and upload.size <= 10 * 1024 * 1024 else b""
        if upload is not None and not content:
            st.warning("Choose a nonempty image of at most 10 MiB.")
        identity = studio.fingerprint([loaded.metadata if loaded else None, hashlib.sha256(content).hexdigest(), candidates, target])
        if st.button("Explain image", type="primary", disabled=loaded is None or not content or not 2 <= len(candidates) <= 16):
            try:
                with studio_jobs.task("image", "Explain image"):
                    record, display, heatmap = studio.explain_image(loaded, content, candidates, target, progress=_progress("image"))
                _save(identity, record, name, display=display[:224, :224], heatmap=heatmap[:224, :224])
            except Exception as exc:
                studio_jobs.failure("image", studio.safe_error(exc))
                st.error(studio.safe_error(exc))
    studio_jobs.render(kind)
    if loaded is None:
        st.info("Compose your input here, then open the Model setup tab to prepare the selected model. Recorded pipelines can be viewed without loading a model.")
    return identity


def _explain(loaded, kind):
    input_column, output_column = st.columns([1, 1.25], gap="large")
    with input_column, st.container(border=True):
        identity = _compose(loaded, kind)
    with output_column, st.container(border=True):
        st.subheader("Output & explanation")
        saved = st.session_state.get("studio_result")
        if saved:
            if saved["identity"] != identity:
                st.warning("Previous completed result · inputs/settings have changed. It is not a result for the current draft; run explicitly to replace it.")
            studio_views.results(saved)
        else:
            st.info("Your model output and attribution will appear here after an explicit run.")
            st.caption("Nothing has been generated yet. For existing input/output pairs, open Recorded pipelines in the sidebar.")
    studio_views.history()


def _models():
    from autoexplain import studio
    from autoexplain import studio_example_runner as runner

    st.title("Models")
    st.caption("Choose a supported workflow. Preparation is explicit; inference is local. No arbitrary model code.")
    for model_id, spec in studio.LIBRARY.items():
        with st.container(border=True):
            title, action = st.columns([3, 1])
            ready = studio.readiness(model_id)
            title.subheader(model_id.split("/")[-1])
            title.write("On device" if ready["downloaded"] else "Download needed · consent required")
            title.caption(f"Recommended for {spec['kind']} explanation · ≈ {spec['disk_gb']} GB cache · {spec['runtime_gb']}")
            if action.button("Prepare " + spec["kind"], key="model_route_" + spec["kind"], use_container_width=True):
                st.session_state.studio_page = "Text explanation" if spec["kind"] == "text" else "Image explanation"
                st.rerun()
    with st.container(border=True):
        st.subheader("Gemma 4 · recorded & live interventions")
        st.caption("Recorded evidence is always available. Live: pinned on-device weights, ≥18 GiB available RAM and ≥6 GiB free GPU memory. Checks do not load weights; no downloads here.")
        if st.button("Check Gemma readiness"):
            with st.spinner("Checking local backend, cache and fit…"):
                try:
                    st.session_state.gemma_readiness = runner.execute()
                except Exception:
                    st.error("Backend check failed. Open diagnostics or use the recorded result; no weights loaded.")
        ready = st.session_state.get("gemma_readiness")
        if ready:
            st.write("**Ready for a bounded live run**" if ready.get("ready") else "**Not ready · use recorded evidence or complete local prerequisites**")
            with st.expander("Gemma diagnostics"):
                st.json(ready)
        else:
            st.caption("Not checked yet")
        if st.button("Open Gemma pipeline"):
            st.session_state.studio_page = "Recorded pipelines"
            st.rerun()
    with st.container(border=True):
        st.subheader("Activation language · NLA")
        st.caption("Specialist checkpoint workflow with staged model release and separate fit checks. Prepare only after selecting the supported checkpoints.")
        if st.button("Open activation workflow"):
            st.session_state.studio_page = "Activation language"
            st.rerun()


def main():
    st.set_page_config(page_title="AutoExplain Studio", layout="wide", initial_sidebar_state="expanded")
    studio_theme.apply()
    # Detach draft values from Streamlit's unrendered-widget cleanup. Consent
    # and uploads deliberately are not retained by this mechanism.
    draft_keys = {"gemma_donor", "gemma_recipient", "gemma_target", "gemma_foil", "live_gemma_layer",
                  "studio_example_view", "recorded_pair", "recorded_patch", "studio_run_name", "studio_question",
                  "studio_target_choice", "studio_target_text", "studio_budget", "studio_candidates",
                  "nla_prompt", "nla_target", "nla_budget"}
    for key in list(st.session_state):
        if key in draft_keys or key.startswith(("studio_prompt_", "studio_edit_", "studio_revision_", "studio_path_", "studio_device_", "nla_path_", "nla_revision_", "nla_positions_", "nla_edit_")):
            st.session_state[key] = st.session_state[key]
    try:
        installed = version("transformers")
    except PackageNotFoundError:
        installed = "not installed"
    compatible = installed == "4.57.1"
    with st.sidebar:
        st.title("AutoExplain")
        st.caption("LOCAL EXPLANATION STUDIO")
        pages = ["Recorded pipelines", "Models", "Text explanation", "Image explanation", "Activation language", "Saved experiments", "Real CNN images"]
        page = st.session_state.get("studio_page", pages[0])
        if page == "Synthetic CNN tutorial":
            page = "Real CNN images"
            st.session_state.studio_page = page
        if page != "Real CNN images" and "cnn_model" in st.session_state:
            from autoexplain.studio_demo import unload
            unload()
        for destination in pages:
            if st.button(destination, key="nav_" + destination, use_container_width=True,
                         type="primary" if page == destination else "secondary"):
                st.session_state.studio_page = destination
                st.rerun()
        st.divider()
        st.subheader("Session model")
        loaded = st.session_state.get("studio_model")
        if loaded is None:
            st.caption("No text/image model loaded")
        else:
            st.text(loaded.metadata.get("model_id", "Session model"))
            st.caption("Loaded · owned by this browser session")
        if st.button("Unload model", disabled=loaded is None, key="studio_sidebar_unload"):
            _unload()
            st.rerun()
        st.caption("Local inference · downloads only with consent")
        for state in st.session_state.get("studio_tasks", {}).values():
            st.caption(f"{state['label']}: {state['status']}")
        with st.expander("Runtime & privacy"):
            st.caption(f"Studio runtime: Transformers {installed}. Text / image / NLA require 4.57.1; Gemma uses its separate backend.")
            st.caption("NLA models are released between stages. Cache, session history and loaded models are separate. No app authentication: use trusted devices and host access controls; never expose publicly.")
    st.caption("AUTOEXPLAIN / " + page.upper())
    if page == "Saved experiments":
        from autoexplain.studio_storage import render
        render()
        return
    if page == "Models":
        _models()
        return
    if page == "Recorded pipelines":
        from autoexplain.studio_examples import render
        render()
        return
    st.title(page)
    if page in ("Text explanation", "Image explanation", "Activation language") and not compatible:
        st.error("Live text, image and NLA workflows require Transformers 4.57.1. "
                 "The current 5.18 runtime is not validated; other versions are also blocked.")
        st.write("Detected runtime version:")
        st.code(installed, language=None)
        st.info("Use a dedicated environment with Transformers 4.57.1 and the project's matching dependencies, "
                "then restart the studio from that environment. Do not change a shared environment blindly. "
                "Recorded pipelines remain available without importing Transformers or loading weights.")
        st.code('python -m pip install "transformers==4.57.1"', language="bash")
        st.caption("Run this only in the intended compatible environment, resolve dependency conflicts, then restart. "
                   "Matching the version enables these controls; it is not a hardware-fit guarantee.")
        studio_views.history()
        return
    if page == "Activation language":
        # Release the normal model before NLA can prepare any checkpoint.
        if st.session_state.get("studio_model") is not None:
            _unload()
            st.info("Released the text/image model before entering NLA. Cached files and history remain available.")
        from autoexplain.studio_nla import render
        action = next((key for key in ("nla_download", "nla_validate", "nla_tokenize", "nla_run", "nla_compare")
                       if st.session_state.get(key)), None)
        if action:
            with studio_jobs.task("nla", "Activation language"):
                render()
        else:
            render()
        studio_jobs.render("nla")
    elif page == "Real CNN images":
        if st.session_state.get("studio_model") is not None:
            _unload()
        from autoexplain.studio_demo import main as cnn
        cnn()
    else:
        kind = "text" if page == "Text explanation" else "image"
        st.caption("Explain a specific output, change the input, compare the result. No model loads until you choose Prepare model.")
        workspace, setup = st.tabs(["Input & output", "Model setup"])
        with setup, st.container(border=True):
            loaded = _library(kind)
        with workspace:
            _explain(loaded, kind)


if __name__ == "__main__":
    main()
