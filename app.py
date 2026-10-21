"""Local Streamlit photo interface. Run with: streamlit run app.py."""
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import streamlit as st
import torch

from tourist_places.i18n import LANGUAGES, MESSAGES, model_description, model_name, percentage, text
from tourist_places.explain import explain_model
from tourist_places.hosting import ensure_hosted_models
from tourist_places.inference import UploadError, available_checkpoints, decode_upload, predict_candidates
from tourist_places.learning import load_checkpoint
from tourist_places.model_catalog import model_run_name
from tourist_places.uploader import travel_photo_uploader, uploaded_bytes

PROJECT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("TOURIST_MODEL_DIR", str(PROJECT / "runs")))
HOSTED = os.environ.get("TOURIST_HOSTED") == "1"

initial_language = st.session_state.get("language", st.query_params.get("lang", "es"))
if initial_language not in LANGUAGES:
    initial_language = "es"
st.set_page_config(page_title=text("page_title", initial_language), page_icon="📍", layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1180px; padding-top: 2.7rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #174D45; }
.eyebrow { font-size: .76rem; letter-spacing: .20em; font-weight: 700; color: #52786B; margin-bottom: .7rem; }
.hero-title { font-family: Georgia, serif; font-size: clamp(2.4rem, 4vw, 3.8rem); line-height: 1.08; letter-spacing: -.035em; margin: 0 0 .8rem; color: #164E45; }
.hero-copy { color: #66736B; font-size: 1.08rem; max-width: 640px; line-height: 1.65; }
.quiet-note { color: #768179; font-size: .85rem; }
[data-testid="stFileUploader"] { border-radius: 14px; }
[data-testid="stMetric"] { padding: 1rem; background: #fff; border: 1px solid #E2E5DC; border-radius: 12px; }
div.stButton > button[kind="primary"] { border-radius: 9px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(max_entries=2, show_spinner=False)
def cached_model(path, fingerprint):
    # CPU serving leaves the GPU available for ongoing training; no upload is saved.
    torch.set_num_threads(max(1, min(2 if HOSTED else 4, os.cpu_count() or 1)))
    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(path, device)
    return model, checkpoint, device, threading.Lock()


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_hosted_models(root):
    return ensure_hosted_models(PROJECT / "cloud" / "models.json", root)


# Stable widget keys keep the photo, model, and results when labels change.
_, language_column = st.columns([3, 1])
with language_column:
    language = st.segmented_control("Idioma / Language", list(LANGUAGES), format_func=LANGUAGES.get,
                                    default=initial_language, selection_mode="single", required=True, key="language")
language = language or initial_language
if st.query_params.get("lang") != language:
    st.query_params["lang"] = language


def t(key, **values):
    return text(key, language, **values)


def percent(value):
    return percentage(value, language)


def types_label(types):
    return " · ".join(t(item) for item in types)


def show_model_comparison(paths):
    st.subheader(t("model_comparison"))
    st.caption(t("model_comparison_note"))
    comparison = []
    for path in paths:
        approach_key = model_run_name(path) + "_approach"
        scores = {}
        try:
            recorded = json.loads((path.parent / "test_metrics.json").read_text(encoding="utf-8"))
            if isinstance(recorded, dict):
                scores = recorded
        except (OSError, ValueError):
            pass
        comparison.append({
            t("model_column"): model_name(path, language),
            t("approach_column"): t(approach_key) if approach_key in MESSAGES else model_description(path, language),
            t("top1"): percent(scores["top1"]) if isinstance(scores.get("top1"), (int, float)) else t("not_evaluated"),
            t("top5_column"): percent(scores["top5"]) if isinstance(scores.get("top5"), (int, float)) else t("not_evaluated"),
        })
    st.table(comparison)


with st.sidebar:
    st.markdown("### 📍 México")
    st.caption(t("brand"))
    st.divider()
    threshold = st.slider(t("threshold"), 0.1, 0.95, 0.5, 0.05, key="minimum_score")
    st.caption(t("threshold_help"))
    if st.button(t("refresh"), width="stretch", key="refresh"):
        st.rerun()
    st.divider()
    st.markdown("**" + t("research") + "**")
    st.caption(t("research_note"))
    st.caption(t("hosted_privacy" if HOSTED else "privacy"))

st.markdown(f'<div class="eyebrow">{t("eyebrow")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-title">{t("hero")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-copy">{t("intro")}</div>', unsafe_allow_html=True)
st.write("")

if HOSTED:
    try:
        with st.spinner(t("hosted_loading")):
            cached_hosted_models(str(MODEL_ROOT))
    except (OSError, ValueError, RuntimeError):
        logging.exception("Published model setup failed")
        st.error(t("hosted_error"))
        if st.button(t("retry_models")):
            st.rerun()
        st.stop()

checkpoints = available_checkpoints(MODEL_ROOT)
with st.container(border=True):
    st.subheader(t("model_picker_title"))
    st.caption(t("model_order_note"))
    previous_model = st.session_state.get("model_choice")
    model_index = checkpoints.index(previous_model) if previous_model in checkpoints else 0
    # A new locale-specific widget refreshes Streamlit's cached option labels.
    selected = st.selectbox(t("model"), checkpoints, index=model_index, format_func=lambda p: model_name(p, language), disabled=not checkpoints, key=f"model_choice_{language}")
    st.session_state["model_choice"] = selected
    if selected:
        st.write(model_description(selected, language))

resource = None
fingerprint = None
if selected:
    fingerprint = (str(selected.resolve()), selected.stat().st_mtime_ns)
    try:
        with st.spinner(t("loading")):
            resource = cached_model(*fingerprint)
    except Exception:
        logging.exception("Could not load the selected destination model")
        st.error(t("model_error"))
else:
    st.info(t("model_pending"))
    if st.button(t("check_model"), key="check_model"):
        st.rerun()

explore_tab, quality_tab = st.tabs([t("explore_tab"), t("quality_tab")])
with explore_tab:
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        st.subheader(t("start"))
        upload = travel_photo_uploader(language)
        example_paths = [] if HOSTED else sorted(path for path in (PROJECT / "data" / "ui_examples").glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        example = None
        if example_paths:
            examples = [None, *example_paths]
            previous_example = st.session_state.get("example_photo")
            example_index = examples.index(previous_example) if previous_example in examples else 0
            example = st.selectbox(t("example_label"), examples, index=example_index, format_func=lambda path: path.stem if path else t("choose_example"), key=f"example_photo_{language}")
            st.session_state["example_photo"] = example
        image, image_digest = None, None
        if upload or example:
            try:
                data = uploaded_bytes(upload) if upload else example.read_bytes()
                image = decode_upload(data)
                image_digest = hashlib.sha256(data).hexdigest()
                st.image(image, caption=str(upload.get("name", "")) if upload else example.stem + " · " + t("example_caption"), width="stretch")
            except UploadError as error:
                st.error(t(error.code))
            except OSError:
                st.error(t("upload_unreadable"))
        else:
            st.caption(t("photo_tip"))
        analyze = st.button(t("analyze"), type="primary", disabled=image is None or resource is None, width="stretch", key="analyze")
        if analyze:
            try:
                model, checkpoint, device, model_lock = resource
                started = time.perf_counter()
                with st.spinner(t("analyzing")), model_lock:
                    candidates = predict_candidates(model, checkpoint["classes"], image, device)
                st.session_state["prediction"] = {"image_digest": image_digest, "model": fingerprint,
                                                   "candidates": candidates, "seconds": time.perf_counter() - started}
            except Exception:
                logging.exception("Destination inference failed")
                st.error(t("inference_error"))
    with right:
        st.subheader(t("destinations"))
        result = st.session_state.get("prediction")
        if result and result["image_digest"] == image_digest and result["model"] == fingerprint:
            candidates = result["candidates"]
            winner = candidates[0]
            uncertain = winner["score"] < threshold
            if uncertain:
                st.warning(t("uncertain"))
            else:
                st.success(t("candidate"))
            st.markdown("#### " + winner["name"].title())
            st.caption(winner["state"].title() + " · " + types_label(winner["types"]))
            st.metric(t("closest_score"), percent(winner["score"]))
            st.write("")
            for index, candidate in enumerate(candidates):
                label = f"{index + 1}. {candidate['name'].title()} · {candidate['state'].title()}"
                st.progress(candidate["score"], text=f"{label}  —  {percent(candidate['score'])}")
            st.caption(t("score_note"))
            payload = {"status": "uncertain" if uncertain else "candidate", "minimum_score": threshold,
                       "model": selected.parent.name, "language": language, "candidates": candidates}
            st.download_button(t("download"), json.dumps(payload, ensure_ascii=False, indent=2), file_name=t("download_name"), mime="application/json", key="download")
        else:
            with st.container(border=True):
                st.markdown("#### " + t("empty_title"))
                st.write(t("empty_copy"))
                st.caption(t("photo_tip"))

with quality_tab:
    if resource:
        _, checkpoint, _, _ = resource
        explanation = explain_model(checkpoint, language)
        st.subheader(t("quality_selected"))
        st.write(model_name(selected, language))
        st.markdown("#### " + explanation["build_title"])
        st.write(explanation["build_text"])
        photo_steps, learning_steps = st.tabs([t("prediction_flow"), t("training_flow")])
        with photo_steps:
            st.markdown("#### " + explanation["inference_title"])
            st.mermaid_chart(explanation["inference_mermaid"], alt=explanation["inference_alt"])
        with learning_steps:
            st.markdown("#### " + explanation["training_title"])
            st.mermaid_chart(explanation["training_mermaid"], alt=explanation["training_alt"])
        st.caption(explanation["schematic_note"])
        with st.expander(t("training_details")):
            st.write(explanation["training_text"])
            if explanation["epoch_text"]:
                st.caption(explanation["epoch_text"])
        st.caption(explanation["score_note"])
        st.markdown("#### " + t("quality_title"))
        metrics = checkpoint.get("val_metrics", {})
        evaluation_label = t("val_photos")
        test_path = selected.parent / "test_metrics.json"
        if test_path.is_file():
            try:
                metrics = json.loads(test_path.read_text(encoding="utf-8"))
                evaluation_label = t("test_photos")
            except (OSError, ValueError):
                pass
        st.caption(t("quality_note"))
        columns = st.columns(3)
        columns[0].metric(t("coverage"), len(checkpoint["classes"]))
        columns[1].metric(t("top1"), percent(metrics.get("top1", 0)))
        columns[2].metric(evaluation_label, metrics.get("images", 0))
        st.caption(t("recall", score=percent(metrics.get("macro_recall", 0))))
        st.divider()
        show_model_comparison(checkpoints)
        with st.expander(t("collection")):
            st.dataframe([{t("destination_column"): item["name"].title(), t("state_column"): item["state"].title(), t("type_column"): types_label(item["types"])} for item in checkpoint["classes"]], hide_index=True, width="stretch")
    else:
        st.write(t("quality_pending"))
        if checkpoints:
            show_model_comparison(checkpoints)

st.divider()
st.markdown(f'<div class="quiet-note">MÉXICO · {t("brand")} &nbsp; / &nbsp; {t("hosted_footer" if HOSTED else "footer")}</div>', unsafe_allow_html=True)
