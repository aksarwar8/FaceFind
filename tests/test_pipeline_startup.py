import unittest
from types import SimpleNamespace
from unittest.mock import patch

from starlette.requests import Request

from api.routes import get_gallery_status, get_pipeline, health_check
from face_engine.embedder import FaceEmbedder
from face_engine.configuration import (
    DEFAULT_PIPELINE_ID,
    FALLBACK_PIPELINE_ID,
    PIPELINE_LABELS,
    activate_pipeline,
    initialize_pipeline,
)


def make_app():
    return SimpleNamespace(state=SimpleNamespace(
        pipeline_bundles={}, gallery_cache={}, pending_cache={},
        active_pipeline_id=None, detector=None, embedder=None, matcher=None,
        gallery=None, pending_references=[], gallery_errors=[],
        gallery_build_errors=[], gallery_timings={}, startup_error=None,
        fallback_reason=None,
    ))


def bundle(pipeline_id):
    return {
        "id": pipeline_id, "label": PIPELINE_LABELS[pipeline_id],
        "detector": object(), "embedder": object(), "matcher": object(),
        "model_initialization_ms": 1.0,
    }


def request_for(app):
    return Request({"type": "http", "app": app, "headers": []})


class PipelineStartupTests(unittest.TestCase):
    def test_fallback_metadata_exposes_sface_embedding_dimensions(self):
        app = make_app()
        fallback = {**bundle(FALLBACK_PIPELINE_ID), "embedder": FaceEmbedder.__new__(FaceEmbedder)}
        with patch("face_engine.configuration.build_pipeline", side_effect=[FileNotFoundError("SCRFD missing"), fallback]):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        metadata = get_pipeline(request_for(app))
        self.assertEqual(metadata["requested_pipeline_id"], DEFAULT_PIPELINE_ID)
        self.assertEqual(metadata["active_pipeline_id"], FALLBACK_PIPELINE_ID)
        self.assertEqual(metadata["embedding_dimensions"], 128)

    def test_preferred_pipeline_available(self):
        app = make_app()
        with patch("face_engine.configuration.build_pipeline", return_value=bundle(DEFAULT_PIPELINE_ID)):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertEqual(app.state.active_pipeline_id, DEFAULT_PIPELINE_ID)
        self.assertIsNone(app.state.fallback_reason)
        self.assertIsNone(app.state.startup_error)

    def test_preferred_pipeline_missing_falls_back_with_reason(self):
        app = make_app()
        def build(pid):
            if pid == DEFAULT_PIPELINE_ID:
                raise FileNotFoundError("AdaFace checkpoint missing")
            return bundle(pid)
        with patch("face_engine.configuration.build_pipeline", side_effect=build):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertEqual(app.state.active_pipeline_id, FALLBACK_PIPELINE_ID)
        self.assertIn("AdaFace checkpoint missing", app.state.fallback_reason)

    def test_optional_dependency_missing_falls_back(self):
        app = make_app()
        def build(pid):
            if pid == DEFAULT_PIPELINE_ID:
                raise ImportError("No module named onnxruntime")
            return bundle(pid)
        with patch("face_engine.configuration.build_pipeline", side_effect=build):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertEqual(app.state.active_pipeline_id, FALLBACK_PIPELINE_ID)
        self.assertIn("onnxruntime", app.state.fallback_reason)

    def test_invalid_checkpoint_falls_back(self):
        app = make_app()
        def build(pid):
            if pid == DEFAULT_PIPELINE_ID:
                raise ValueError("Invalid ONNX model: failed to load graph")
            return bundle(pid)
        with patch("face_engine.configuration.build_pipeline", side_effect=build):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertEqual(app.state.active_pipeline_id, FALLBACK_PIPELINE_ID)
        self.assertIn("failed to load graph", app.state.fallback_reason)

    def test_fallback_initialization_succeeds_after_preferred_failure(self):
        app = make_app()
        with patch("face_engine.configuration.build_pipeline", side_effect=[RuntimeError("SCRFD unavailable"), bundle(FALLBACK_PIPELINE_ID)]):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertIsNotNone(app.state.detector)
        self.assertEqual(app.state.active_pipeline_id, FALLBACK_PIPELINE_ID)

    def test_explicit_pipeline_selection_is_honored(self):
        app = make_app()
        with patch("face_engine.configuration.build_pipeline", return_value=bundle(FALLBACK_PIPELINE_ID)) as build:
            initialize_pipeline(app, FALLBACK_PIPELINE_ID)
        build.assert_called_once_with(FALLBACK_PIPELINE_ID)
        self.assertEqual(app.state.active_pipeline_id, FALLBACK_PIPELINE_ID)

    def test_explicit_pipeline_failure_reports_alternate_pipeline(self):
        app = make_app()
        def build(pid):
            if pid == FALLBACK_PIPELINE_ID:
                raise FileNotFoundError("YuNet checkpoint missing")
            return bundle(pid)
        with patch("face_engine.configuration.build_pipeline", side_effect=build):
            initialize_pipeline(app, FALLBACK_PIPELINE_ID)
        self.assertEqual(app.state.active_pipeline_id, DEFAULT_PIPELINE_ID)
        self.assertIn("YuNet checkpoint missing", app.state.fallback_reason)

    def test_failed_switch_keeps_active_models_and_gallery(self):
        app = make_app()
        old_bundle = bundle(FALLBACK_PIPELINE_ID)
        old_gallery = {"names": ["A"], "pipeline_id": FALLBACK_PIPELINE_ID}
        app.state.pipeline_bundles[FALLBACK_PIPELINE_ID] = old_bundle
        app.state.active_pipeline_id = FALLBACK_PIPELINE_ID
        app.state.detector, app.state.embedder, app.state.matcher = old_bundle["detector"], old_bundle["embedder"], old_bundle["matcher"]
        app.state.gallery = old_gallery
        with patch("face_engine.configuration.build_pipeline", side_effect=RuntimeError("corrupt checkpoint")):
            with self.assertRaisesRegex(RuntimeError, "corrupt checkpoint"):
                activate_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertEqual(app.state.active_pipeline_id, FALLBACK_PIPELINE_ID)
        self.assertIs(app.state.gallery, old_gallery)
        self.assertIs(app.state.detector, old_bundle["detector"])

    def test_switch_restores_only_gallery_for_selected_pipeline(self):
        app = make_app()
        yunet_bundle = bundle(FALLBACK_PIPELINE_ID)
        research_bundle = bundle(DEFAULT_PIPELINE_ID)
        app.state.pipeline_bundles.update({FALLBACK_PIPELINE_ID: yunet_bundle, DEFAULT_PIPELINE_ID: research_bundle})
        app.state.gallery_cache[FALLBACK_PIPELINE_ID] = {"names": ["Y"], "pipeline_id": FALLBACK_PIPELINE_ID}
        app.state.gallery_cache[DEFAULT_PIPELINE_ID] = {"names": ["S"], "pipeline_id": DEFAULT_PIPELINE_ID}
        app.state.pending_cache[DEFAULT_PIPELINE_ID] = [{"pending_id": 0}]
        activate_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertEqual(app.state.gallery["names"], ["S"])
        self.assertEqual(app.state.pending_references, [{"pending_id": 0}])
        activate_pipeline(app, FALLBACK_PIPELINE_ID)
        self.assertEqual(app.state.gallery["names"], ["Y"])
        self.assertEqual(app.state.pending_references, [])

    def test_mismatched_cached_gallery_is_never_activated(self):
        app = make_app()
        app.state.pipeline_bundles[FALLBACK_PIPELINE_ID] = bundle(FALLBACK_PIPELINE_ID)
        app.state.gallery_cache[FALLBACK_PIPELINE_ID] = {
            "names": ["AdaFace embedding"], "pipeline_id": DEFAULT_PIPELINE_ID,
        }
        app.state.pending_cache[FALLBACK_PIPELINE_ID] = [{"pending_id": 0}]
        activate_pipeline(app, FALLBACK_PIPELINE_ID)
        self.assertIsNone(app.state.gallery)
        self.assertEqual(app.state.pending_references, [])
        self.assertNotIn(FALLBACK_PIPELINE_ID, app.state.gallery_cache)

    def test_complete_initialization_failure_leaves_engine_unavailable(self):
        app = make_app()
        with patch("face_engine.configuration.build_pipeline", side_effect=RuntimeError("model load failed")):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        self.assertIsNone(app.state.active_pipeline_id)
        self.assertIsNone(app.state.detector)
        self.assertIn("model load failed", app.state.startup_error)

    def test_health_and_gallery_status_agree_when_no_pipeline_loads(self):
        app = make_app()
        with patch("face_engine.configuration.build_pipeline", side_effect=RuntimeError("unavailable")):
            initialize_pipeline(app, DEFAULT_PIPELINE_ID)
        request = request_for(app)
        health = health_check(request)
        status = get_gallery_status(request)
        self.assertFalse(health["engine_initialized"])
        self.assertEqual(health["status"], "unavailable")
        self.assertFalse(status["engine_initialized"])
        self.assertEqual(health["startup_error"], status["startup_error"])
        self.assertIsNone(status["active_pipeline_label"])

    def test_health_and_gallery_status_agree_when_pipeline_is_ready(self):
        app = make_app()
        with patch("face_engine.configuration.build_pipeline", return_value=bundle(FALLBACK_PIPELINE_ID)):
            initialize_pipeline(app, FALLBACK_PIPELINE_ID)
        request = request_for(app)
        health = health_check(request)
        status = get_gallery_status(request)
        self.assertTrue(health["engine_initialized"])
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(status["engine_initialized"])
        self.assertEqual(health["active_pipeline_id"], status["active_pipeline_id"])


if __name__ == "__main__":
    unittest.main()
