import asyncio
import base64
import hashlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from starlette.datastructures import Headers, UploadFile
from starlette.requests import Request

from fastapi import HTTPException

from api.routes import (AutomaticFaceCorrection, FaceSelection, analyze_group_photo,
                        correct_automatic_reference, get_automatic_reference,
                        get_pending_reference, PendingAction, select_reference_face, skip_reference, upload_gallery)
from face_engine.match import FaceMatcher
from face_engine.reference_selection import dominant_face_selection


def encoded_image(value):
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def upload(filename, payload):
    return UploadFile(
        filename=filename,
        file=io.BytesIO(payload),
        headers=Headers({"content-type": "image/jpeg"}),
    )


class FakeDetector:
    def detect(self, image):
        level = int(round(float(image.mean())))
        if level == 3:
            return []
        count = 2 if level == 2 else 1
        return [
            {
                "bbox": [index, index, 8, 8],
                "landmarks": [1, 1, 5, 1, 3, 3, 2, 5, 4, 5],
                "confidence": 0.99,
            }
            for index in range(count)
        ]


class FakeEmbedder:
    dimension = 2

    def __init__(self, vector=None):
        self.vector = np.asarray(vector or [1.0, 0.0], dtype=np.float32)

    def get_embedding(self, image, detection):
        self.last_timings_ms = {"alignment": 0.1, "embedding": 0.2}
        return self.vector


def make_request():
    state = SimpleNamespace(
        detector=FakeDetector(), embedder=FakeEmbedder(), matcher=FaceMatcher(),
        active_pipeline_id="yunet_sface", pipeline_bundles={
            "yunet_sface": {"label": "YuNet + SFace", "model_initialization_ms": 1.0}
        }, gallery=None, pending_references=[], gallery_cache={}, pending_cache={},
        gallery_errors=[], gallery_build_errors=[], gallery_timings={},
        startup_error=None, fallback_reason=None,
    )
    app = SimpleNamespace(state=state)
    return Request({"type": "http", "app": app, "headers": []})


class GalleryApiLifecycleTests(unittest.TestCase):
    def test_pending_reference_cannot_enroll_a_case_variant_of_an_existing_identity(self):
        request = make_request()
        asyncio.run(upload_gallery(request, [upload("Alice.jpg", encoded_image(1))]))
        followup = asyncio.run(upload_gallery(request, [upload("ALICE.jpg", encoded_image(2))]))
        self.assertEqual(followup["pending_references"], 1)
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as error:
            select_reference_face(FaceSelection(pending_id=request.app.state.pending_references[0]["pending_id"], face_index=0), request)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(request.app.state.gallery["names"], ["Alice"])

    def test_dominant_face_requires_area_and_a_unique_highest_score(self):
        def face(width, confidence):
            return {"bbox": [0, 0, width, 1], "confidence": confidence}

        small = face(2, 0.74)
        dominant = face(10, 0.75)
        evidence = dominant_face_selection([small, dominant, face(1, 0.50)])
        self.assertEqual(evidence, {"reason": "dominant_face", "selected_face_index": 1,
                                    "area_ratio": 5.0, "detection_score": 0.75})
        for other_faces in ([face(2.01, 0.74), dominant], [small, face(10, 0.749)],
                            [face(2, 0.76), dominant], [face(2, 0.75), dominant],
                            [small, dominant, face(1, 0.76)],
                            [face(0, 0.74), dominant], [face(2, float("nan")), dominant]):
            with self.subTest(faces=other_faces):
                self.assertIsNone(dominant_face_selection(other_faces))
        self.assertIsNone(dominant_face_selection([dominant]))
        self.assertIsNone(dominant_face_selection([small, dominant], "always_review"))

    def test_automatic_enrollment_can_be_corrected_without_reordering_gallery(self):
        request = make_request()
        state = request.app.state
        small = {"bbox": [10, 10, 2, 2], "confidence": 0.55}
        large = {"bbox": [0, 0, 8, 8], "confidence": 0.85}
        state.detector.detect = Mock(side_effect=[[small, large], [large]])
        state.embedder.get_embedding = Mock(side_effect=[[1, 0], [1, 1], [0, 2]])
        result = asyncio.run(upload_gallery(request, [
            upload("Target.jpg", encoded_image(2)), upload("Other.jpg", encoded_image(1)),
        ]))
        self.assertEqual(result["identities"], ["Target", "Other"])
        self.assertFalse(result["pending"])
        self.assertEqual(result["upload_summary"]["automatic_selection_references"], 1)
        review = result["automatic_references"][0]
        self.assertEqual(review["automatic_selection"]["selected_face_index"], 1)
        self.assertEqual(review["selected_face_index"], 1)
        self.assertIs(state.embedder.get_embedding.call_args_list[0].args[1], large)
        revision = result["gallery_revision"]
        preview = get_automatic_reference(revision, review["pending_id"], request)
        self.assertEqual([face["confidence"] for face in preview["candidates"]], [0.55, 0.85])
        saved = correct_automatic_reference(review["pending_id"], AutomaticFaceCorrection(revision=revision, face_index=1), request)
        self.assertEqual(saved["gallery_revision"], revision)
        original_other = state.gallery["embeddings"][1].copy()
        changed = correct_automatic_reference(review["pending_id"], AutomaticFaceCorrection(revision=revision, face_index=0), request)
        self.assertEqual(changed["identities"], result["identities"])
        self.assertNotEqual(changed["gallery_revision"], revision)
        self.assertEqual(changed["automatic_references"][0]["selected_face_index"], 0)
        self.assertEqual(changed["automatic_references"][0]["automatic_selection"], review["automatic_selection"])
        np.testing.assert_array_equal(state.gallery["embeddings"][0], [0, 1])
        np.testing.assert_array_equal(state.gallery["embeddings"][1], original_other)
        self.assertIs(state.embedder.get_embedding.call_args.args[1], small)
        with self.assertRaises(HTTPException) as stale:
            correct_automatic_reference(review["pending_id"], AutomaticFaceCorrection(revision=revision, face_index=1), request)
        self.assertEqual(stale.exception.status_code, 409)
        state.detector.detect.side_effect = None
        state.detector.detect.return_value = [small, large]
        state.embedder.get_embedding = FakeEmbedder().get_embedding
        analysis = asyncio.run(analyze_group_photo(request, upload("group.jpg", encoded_image(2))))
        self.assertEqual(analysis["face_count"], 2)

    def test_explicit_review_policy_queues_even_a_dominant_face(self):
        request = make_request()
        state = request.app.state
        state.detector.detect = Mock(return_value=[
            {"bbox": [0, 0, 8, 8], "confidence": 0.85},
            {"bbox": [10, 10, 2, 2], "confidence": 0.55},
        ])
        state.embedder.get_embedding = Mock(wraps=state.embedder.get_embedding)
        result = asyncio.run(upload_gallery(request, [
            upload("Target.jpg", encoded_image(2)), upload("Next.jpg", encoded_image(2)),
        ], "always_review"))
        first, second = result["pending"]
        self.assertEqual([first["faces"], second["faces"]], [2, 2])
        self.assertFalse(result["automatic_references"])
        state.embedder.get_embedding.assert_not_called()
        self.assertIsInstance(first["pending_id"], str)
        self.assertNotEqual(first["pending_id"], second["pending_id"])

        preview = get_pending_reference(first["pending_id"], request, interactive=True)
        self.assertEqual(preview["pending_id"], first["pending_id"])
        self.assertEqual(preview["filename"], first["filename"])
        self.assertEqual([candidate["index"] for candidate in preview["candidates"]], [0, 1])
        self.assertTrue(all(candidate["crop"].startswith("data:image/jpeg;base64,")
                            for candidate in preview["candidates"]))
        updated = select_reference_face(FaceSelection(pending_id=first["pending_id"], face_index=1), request)
        self.assertEqual(updated["people_count"], 1)
        self.assertEqual(updated["pending"], [second])
        self.assertEqual(get_pending_reference(second["pending_id"], request, interactive=True)["pending_id"],
                         second["pending_id"])

    def test_invalid_box_does_not_send_a_single_valid_reference_face_to_review(self):
        request = make_request()
        state = request.app.state
        valid = {"bbox": [0, 0, 8, 8], "confidence": 0.70}
        state.detector.detect = Mock(return_value=[{"bbox": [10, 10, 0, 2], "confidence": 0.85}, valid])
        state.embedder.get_embedding = Mock(wraps=state.embedder.get_embedding)
        result = asyncio.run(upload_gallery(request, [upload("Single.jpg", encoded_image(1))], "always_review"))
        self.assertEqual(result["identities"], ["Single"])
        self.assertFalse(result["pending"])
        self.assertIs(state.embedder.get_embedding.call_args.args[1], valid)

    def test_interactive_reference_preview_crops_and_selects_the_original_indices(self):
        request = make_request()
        state = request.app.state
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100] = [0, 0, 255]
        image[:, 100:] = [255, 0, 0]
        faces = [
            {"bbox": [-3, 20, 30, 40], "confidence": 0.91},
            {"bbox": [130, 20, 50, 50], "confidence": 0.54},
        ]
        state.pending_references = [{
            "pending_id": "review-item", "identity": "Target", "filename": "Target.jpg", "image": image, "faces": faces,
        }]
        state.gallery = {"names": [], "embeddings": np.empty((0, 2), np.float32)}

        preview = get_pending_reference("review-item", request, interactive=True)
        self.assertEqual((preview["width"], preview["height"]), (200, 100))
        candidates = preview["candidates"]
        self.assertEqual([face["index"] for face in candidates], [0, 1])
        self.assertEqual([face["confidence"] for face in candidates], [0.91, 0.54])
        self.assertEqual(candidates[0]["bbox"], [0, 20, 27, 40])
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(candidates[1]["crop"].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(crop[:, :, 0].mean(), 250)
        self.assertNotIn("candidates", get_pending_reference("review-item", request))

        state.embedder.get_embedding = Mock(wraps=state.embedder.get_embedding)
        select_reference_face(FaceSelection(pending_id=request.app.state.pending_references[0]["pending_id"], face_index=1), request)
        selected_image, selected_face = state.embedder.get_embedding.call_args.args
        self.assertIs(selected_image, image)
        self.assertIs(selected_face, faces[1])

    def test_scrfd_retry_enrolls_single_faces_and_keeps_ambiguous_and_empty_outcomes(self):
        from face_engine.research_models import StridePaddedSCRFD

        request = make_request()
        state = request.app.state
        state.active_pipeline_id = "scrfd_adaface"
        state.detector = StridePaddedSCRFD.__new__(StridePaddedSCRFD)
        face = {"bbox": [1, 1, 8, 8], "landmarks": [2, 2, 7, 2, 4, 4, 2, 6, 7, 6], "confidence": 0.9}
        state.detector.adapter = Mock()
        state.detector.adapter.detect.side_effect = [[], [face], [], [face, face], [], []]
        state.embedder.get_embedding = Mock(wraps=state.embedder.get_embedding)
        result = asyncio.run(upload_gallery(request, [
            upload("Single.jpg", encoded_image(1)), upload("Multiple.jpg", encoded_image(2)),
            upload("None.jpg", encoded_image(3)),
        ]))
        summary = result["upload_summary"]
        self.assertEqual(state.gallery["names"], ["Single"])
        self.assertEqual(result["pending"][0]["faces"], 2)
        self.assertEqual(summary["gallery_detection_retry_count"], 3)
        self.assertEqual(summary["gallery_detection_retry_recovered_count"], 2)
        self.assertEqual([row["detection"]["pass"] for row in summary["per_reference"]], ["retry_640"] * 3)
        self.assertIn("either SCRFD scale", summary["per_reference"][2]["error"])
        state.embedder.get_embedding.assert_called_once()
        self.assertEqual(state.embedder.get_embedding.call_args.args[0].shape, (16, 16, 3))

        # Analyze the same model bundle with no detections; there is no second pass.
        state.detector.adapter.detect.reset_mock(side_effect=True)
        state.detector.adapter.detect.return_value = []
        analysis = asyncio.run(analyze_group_photo(request, upload("group.jpg", encoded_image(3))))
        self.assertEqual(analysis["face_count"], 0)
        state.detector.adapter.detect.assert_called_once()

    def test_invalid_no_face_and_multi_face_refs_are_reported_and_selectable(self):
        request = make_request()
        files = [
            upload("Alice.jpg", encoded_image(1)),
            upload("Bob.jpg", encoded_image(2)),
            upload("NoFace.jpg", encoded_image(3)),
            upload("Broken.jpg", b"not an image"),
            upload("Alice.png", encoded_image(1)),
            upload("notes.txt", b"ignored"),
        ]
        with patch("api.routes.time.perf_counter", side_effect=iter([1.0, 1.1] + [1.2] * 20)):
            result = asyncio.run(upload_gallery(request, files))

        self.assertTrue(result["gallery_loaded"])
        self.assertEqual(request.app.state.gallery["names"], ["Alice"])
        self.assertEqual(result["upload_summary"]["accepted_references"], 1)
        self.assertEqual(result["upload_summary"]["manual_selection_references"], 1)
        self.assertEqual(result["upload_summary"]["rejected_references"], 4)
        self.assertEqual(request.app.state.pending_references[0]["identity"], "Bob")

        selected = select_reference_face(
            FaceSelection(pending_id=request.app.state.pending_references[0]["pending_id"], face_index=1), request
        )
        self.assertEqual(request.app.state.gallery["names"], ["Alice", "Bob"])
        self.assertEqual(selected["pending_references"], 0)
        self.assertEqual(selected["people_count"], 2)

    def test_failed_gallery_replacement_keeps_working_gallery(self):
        request = make_request()
        gallery = {"names": ["Existing"], "embeddings": np.array([[1.0, 0.0]], np.float32),
                   "errors": [], "pending": [], "pipeline_id": "yunet_sface"}
        request.app.state.gallery = gallery
        request.app.state.gallery_cache["yunet_sface"] = gallery
        result = asyncio.run(upload_gallery(request, [upload("bad.tiff", b"ignored")]))
        self.assertFalse(result["build_succeeded"])
        self.assertIs(request.app.state.gallery, gallery)
        self.assertEqual(request.app.state.gallery["names"], ["Existing"])

    def test_unknown_face_is_not_returned_with_a_gallery_identity(self):
        request = make_request()
        request.app.state.embedder = FakeEmbedder([0.0, 1.0])
        request.app.state.gallery = {
            "names": ["Known"], "embeddings": np.array([[1.0, 0.0]], np.float32),
            "pipeline_id": "yunet_sface",
        }
        payload = encoded_image(1)
        response = asyncio.run(analyze_group_photo(request, upload("unknown.jpg", payload)))
        self.assertEqual(response["summary"]["unknown"], 1)
        self.assertIsNone(response["results"][0]["identity"])
        self.assertRegex(response["analysis_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(response["image_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(response["image_geometry"], {"original_width": 16, "original_height": 16, "orientation": 1})
        self.assertEqual(response["gallery_identities"], ["Known"])
        self.assertEqual(response["recognition_rules"]["recognition_threshold"],
                         request.app.state.matcher.recognition_threshold)

    def test_stale_review_does_not_select_same_stem_after_queue_advances(self):
        request = make_request()
        asyncio.run(upload_gallery(request, [upload("Existing.jpg", encoded_image(1))]))
        initial = asyncio.run(upload_gallery(request, [upload("Same.jpg", encoded_image(2))]))
        old_id = initial["pending"][0]["pending_id"]
        second = asyncio.run(upload_gallery(request, [
            upload("Same.png", encoded_image(2)), upload("Similar.jpg", encoded_image(2)),
        ]))
        next_id, similar_id = [item["pending_id"] for item in second["pending"][1:]]
        self.assertEqual(len({old_id, next_id, similar_id}), 3)
        stale_selection = FaceSelection(pending_id=old_id, face_index=0, expected_identity="Same")
        skipped = skip_reference(PendingAction(pending_id=old_id), request)
        self.assertEqual(skipped["gallery_revision"], initial["gallery_revision"])
        for action in (
            lambda: select_reference_face(stale_selection, request),
            lambda: get_pending_reference(old_id, request),
            lambda: skip_reference(PendingAction(pending_id=old_id), request),
        ):
            with self.assertRaises(HTTPException) as error:
                action()
            self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(get_pending_reference(next_id, request)["filename"], "Same.png")
        selection = FaceSelection(pending_id=next_id, face_index=0, expected_identity="Same")
        selected = select_reference_face(selection, request)
        with self.assertRaises(HTTPException) as error:
            select_reference_face(selection, request)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(selected["pending"][0]["pending_id"], similar_id)

    def test_pending_id_cannot_select_after_switch_or_replacement(self):
        request = make_request()
        state = request.app.state
        result = asyncio.run(upload_gallery(request, [
            upload("Existing.jpg", encoded_image(1)), upload("Same.jpg", encoded_image(2)),
        ]))
        old_id = result["pending"][0]["pending_id"]
        selection = FaceSelection(pending_id=old_id, face_index=0, expected_identity="Same")
        state.active_pipeline_id = "scrfd_adaface"
        state.gallery = None
        state.pending_references = []
        with self.assertRaisesRegex(HTTPException, "stale"):
            select_reference_face(selection, request)
        state.active_pipeline_id = "yunet_sface"
        state.gallery = state.gallery_cache["yunet_sface"]
        state.pending_references = state.pending_cache["yunet_sface"]
        self.assertEqual(get_pending_reference(old_id, request)["filename"], "Same.jpg")
        replacement = asyncio.run(upload_gallery(request, [
            upload("Other.jpg", encoded_image(1)), upload("Same.png", encoded_image(2)),
        ]))
        self.assertNotEqual(replacement["pending"][0]["pending_id"], old_id)
        with self.assertRaisesRegex(HTTPException, "stale"):
            select_reference_face(selection, request)


if __name__ == "__main__":
    unittest.main()
