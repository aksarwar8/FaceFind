"""Offline selectable detector/recognizer configurations for the local demo."""
from __future__ import annotations

from pathlib import Path
import re
import time

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.match import FaceMatcher


PIPELINE_LABELS = {
    "scrfd_adaface": "SCRFD-10G-KPS + AdaFace IR-18 (research demo)",
    "yunet_sface": "YuNet + SFace (fallback)",
}

DEFAULT_PIPELINE_ID = "scrfd_adaface"
FALLBACK_PIPELINE_ID = "yunet_sface"


def model_error_detail(error):
    """Keep actionable model errors while removing private local paths."""
    root = Path(__file__).resolve().parent.parent
    detail = str(error).replace(str(root), ".").replace(root.as_posix(), ".")
    # Native libraries may include build paths and checkpoint paths outside the checkout.
    detail = re.sub(r"""(?i)(?:[a-z]:[/\\]|\\\\)[^"'\r\n<>|]*""", "[local path]", detail)
    detail = re.sub(r"""(?<![\w:/])/(?:[^/"'\r\n]+/)+[^"'\r\n]*""", "[local path]", detail)
    return " ".join(detail.split())[:500]


def build_pipeline(pipeline_id):
    if pipeline_id not in PIPELINE_LABELS:
        raise ValueError(f"Unknown pipeline configuration: {pipeline_id}")
    started = time.perf_counter()
    if pipeline_id == "scrfd_adaface":
        from face_engine.research_models import AdaFaceEmbedder, AdaFaceGalleryMatcher, StridePaddedSCRFD
        detector, embedder, matcher = StridePaddedSCRFD(), AdaFaceEmbedder(), AdaFaceGalleryMatcher()
    else:
        detector, embedder, matcher = FaceDetector(), FaceEmbedder(), FaceMatcher()
    elapsed = (time.perf_counter() - started) * 1000
    for component in (detector, embedder, matcher):
        if not hasattr(component, "cache_key"):
            component.cache_key = pipeline_id
    return {
        "id": pipeline_id,
        "label": PIPELINE_LABELS[pipeline_id],
        "detector": detector,
        "embedder": embedder,
        "matcher": matcher,
        "model_initialization_ms": elapsed,
    }


def initialize_pipeline(app, requested_pipeline_id):
    """Initialize the requested pipeline, falling back with an explicit reason.

    The app remains available for its health/status endpoints even when neither
    local model bundle can be initialized.
    """
    if requested_pipeline_id not in PIPELINE_LABELS:
        requested_pipeline_id = DEFAULT_PIPELINE_ID
        request_note = "Unrecognized FACEFIND_DEFAULT_PIPELINE; using preferred pipeline."
    else:
        request_note = None

    app.state.requested_pipeline_id = requested_pipeline_id
    try:
        activate_pipeline(app, requested_pipeline_id)
        app.state.startup_error = None
        app.state.fallback_reason = request_note
        return
    except Exception as error:
        requested_error = f"{PIPELINE_LABELS[requested_pipeline_id]} unavailable: {model_error_detail(error)}"

    fallback_pipeline_id = (
        FALLBACK_PIPELINE_ID if requested_pipeline_id == DEFAULT_PIPELINE_ID
        else DEFAULT_PIPELINE_ID
    )
    try:
        activate_pipeline(app, fallback_pipeline_id)
        app.state.startup_error = None
        app.state.fallback_reason = requested_error
    except Exception as fallback_error:
        app.state.active_pipeline_id = None
        app.state.detector = None
        app.state.embedder = None
        app.state.matcher = None
        app.state.gallery = None
        app.state.pending_references = []
        app.state.startup_error = (
            f"{requested_error}; {PIPELINE_LABELS[fallback_pipeline_id]} also unavailable: "
            f"{model_error_detail(fallback_error)}"
        )
        app.state.fallback_reason = app.state.startup_error


def activate_pipeline(app, pipeline_id):
    if pipeline_id not in PIPELINE_LABELS:
        raise ValueError(f"Unknown pipeline configuration: {pipeline_id}")
    bundle = app.state.pipeline_bundles.get(pipeline_id)
    if bundle is None:
        bundle = build_pipeline(pipeline_id)
        app.state.pipeline_bundles[pipeline_id] = bundle
    cached_gallery = app.state.gallery_cache.get(pipeline_id)
    if cached_gallery and cached_gallery.get("pipeline_id") != pipeline_id:
        # Never attach embeddings produced by another model configuration.
        app.state.gallery_cache.pop(pipeline_id, None)
        app.state.pending_cache.pop(pipeline_id, None)
        cached_gallery = None
    app.state.active_pipeline_id = pipeline_id
    app.state.detector = bundle["detector"]
    app.state.embedder = bundle["embedder"]
    app.state.matcher = bundle["matcher"]
    app.state.gallery = cached_gallery
    app.state.pending_references = app.state.pending_cache.get(pipeline_id, [])
    app.state.gallery_errors = (app.state.gallery or {}).get("errors", [])
    app.state.gallery_build_errors = (app.state.gallery or {}).get("errors", [])
    app.state.gallery_timings = (app.state.gallery or {}).get("timings", {})
