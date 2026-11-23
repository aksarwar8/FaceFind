"""Offline selectable detector/recognizer configurations for the local demo."""
from __future__ import annotations

import time

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.match import FaceMatcher


PIPELINE_LABELS = {
    "scrfd_adaface": "SCRFD-10G-KPS + AdaFace IR-18 (research demo)",
    "yunet_sface": "YuNet + SFace (fallback)",
}


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


def activate_pipeline(app, pipeline_id):
    bundle = app.state.pipeline_bundles.get(pipeline_id)
    if bundle is None:
        bundle = build_pipeline(pipeline_id)
        app.state.pipeline_bundles[pipeline_id] = bundle
    app.state.active_pipeline_id = pipeline_id
    app.state.detector = bundle["detector"]
    app.state.embedder = bundle["embedder"]
    app.state.matcher = bundle["matcher"]
    app.state.gallery = app.state.gallery_cache.get(pipeline_id)
    app.state.pending_references = app.state.pending_cache.get(pipeline_id, [])
    app.state.gallery_errors = (app.state.gallery or {}).get("errors", [])
