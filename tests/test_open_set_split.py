"""Focused tests for safe open-set identity and gallery-reference selection."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts.evaluation import evaluate_open_set
from scripts.evaluation.evaluate_open_set import (
    evaluate,
    make_split,
    preflight_gallery_reference,
)


class OpenSetSplitTests(unittest.TestCase):
    def test_invalid_primary_identity_is_replaced_from_seeded_reserve(self):
        identities = {
            f"person_{index:02d}": [
                Path(f"person_{index:02d}_01.jpg"),
                Path(f"person_{index:02d}_02.jpg"),
            ]
            for index in range(12)
        }
        seed = 19
        shuffled = sorted(identities)
        random.Random(seed).shuffle(shuffled)
        primary = shuffled[:3]
        unknown_expected = shuffled[3:5]
        reserve_expected = shuffled[5:]
        rejected_identity = primary[1]

        def preflight(identity, path):
            return {
                "identity": identity,
                "reference_image": str(path),
                "accepted": identity != rejected_identity,
                "reason": None if identity != rejected_identity else "multiple_faces_detected",
                "face_count": 1 if identity != rejected_identity else 2,
            }

        known, unknown, audit = make_split(
            identities, preflight, seed=seed, known_count=3, unknown_count=2
        )

        self.assertEqual(set(known), {primary[0], primary[2], reserve_expected[0]})
        self.assertEqual(list(unknown), unknown_expected)
        self.assertEqual(set(known).intersection(unknown), set())
        self.assertEqual(audit["excluded_identities"][0]["identity"], rejected_identity)
        self.assertEqual(audit["reserve_replacements"][0]["identity"], reserve_expected[0])
        self.assertEqual(
            audit["reserve_replacements"][0]["replaces_identity"], rejected_identity
        )
        for split in known.values():
            self.assertNotIn(split["gallery"], split["probes"])

    def test_raises_if_reserve_cannot_fill_requested_known_gallery(self):
        identities = {
            f"person_{index:02d}": [Path(f"{index}_1.jpg"), Path(f"{index}_2.jpg")]
            for index in range(8)
        }

        def reject_all(identity, path):
            return {"identity": identity, "accepted": False, "reason": "multiple_faces_detected"}

        with self.assertRaisesRegex(ValueError, "Only 0 known identities"):
            make_split(identities, reject_all, seed=4, known_count=2, unknown_count=2)


class GalleryReferencePreflightTests(unittest.TestCase):
    def setUp(self):
        self.detection = {
            "bbox": [1, 2, 30, 40],
            "landmarks": [float(value) for value in range(10)],
            "confidence": 0.95,
        }
        self.image = object()

    @patch("scripts.evaluation.evaluate_open_set.cv2.imread")
    def test_accepts_exactly_one_usable_face_and_embedding(self, imread):
        imread.return_value = self.image
        detector = Mock()
        detector.detect.return_value = [self.detection]
        embedder = Mock()
        embedder.get_embedding.return_value = [0.5, 0.5]

        result = preflight_gallery_reference("Ada", Path("ada.jpg"), detector, embedder)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["face_count"], 1)
        self.assertEqual(result["detection"]["bbox"], self.detection["bbox"])

    @patch("scripts.evaluation.evaluate_open_set.cv2.imread")
    def test_rejects_zero_and_multiple_faces_without_selecting_one(self, imread):
        imread.return_value = self.image
        for detections, reason in (([], "no_face_detected"),
                                   ([self.detection, self.detection], "multiple_faces_detected")):
            detector = Mock()
            detector.detect.return_value = detections
            embedder = Mock()

            result = preflight_gallery_reference("Ada", Path("ada.jpg"), detector, embedder)

            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], reason)
            embedder.get_embedding.assert_not_called()

    @patch("scripts.evaluation.evaluate_open_set.cv2.imread")
    def test_rejects_invalid_detection_and_invalid_embedding(self, imread):
        imread.return_value = self.image
        invalid = {**self.detection, "bbox": [1, 2, -3, 40]}
        detector = Mock()
        detector.detect.return_value = [invalid]
        result = preflight_gallery_reference("Ada", Path("ada.jpg"), detector, Mock())
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "invalid_face_detection")

        detector.detect.return_value = [self.detection]
        embedder = Mock()
        embedder.get_embedding.return_value = [0.0, 0.0]
        result = preflight_gallery_reference("Ada", Path("ada.jpg"), detector, embedder)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "gallery_embedding_error")

    @patch("scripts.evaluation.evaluate_open_set.cv2.imread", side_effect=OSError("decoder failed"))
    def test_image_read_exception_is_recorded_and_reserve_replacement_continues(self, imread):
        detector = Mock()
        detector.detect.return_value = [self.detection]
        embedder = Mock()
        embedder.get_embedding.return_value = [0.5, 0.5]
        identities = {
            f"person_{index}": [Path(f"person_{index}_1.jpg"), Path(f"person_{index}_2.jpg")]
            for index in range(4)
        }
        calls = 0

        def preflight(identity, path):
            nonlocal calls
            calls += 1
            if calls == 1:
                return preflight_gallery_reference(identity, path, detector, embedder)
            with patch("scripts.evaluation.evaluate_open_set.cv2.imread", return_value=self.image):
                return preflight_gallery_reference(identity, path, detector, embedder)

        known, unknown, audit = make_split(
            identities, preflight, seed=4, known_count=1, unknown_count=1
        )

        self.assertEqual(len(known), 1)
        self.assertEqual(len(unknown), 1)
        self.assertEqual(audit["excluded_identities"][0]["reason"], "gallery_image_read_error")
        self.assertEqual(audit["excluded_identities"][0]["detail"], "decoder failed")


class TransactionalOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp_context = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_context.name)
        self.output_root = self.root / "evaluation"
        self.output_root.mkdir()
        self.final_dir = self.output_root / "open_set_test"
        self.image_root = self.root / "lfw"
        self.image_root.mkdir()

    def tearDown(self):
        self.temp_context.cleanup()

    def _patch_run_dependencies(self, gallery_side_effect=None):
        known = {"known_person": {"gallery": Path("known_person_1.jpg"), "probes": [Path("known_person_2.jpg")]}}
        unknown = {"unknown_person": [Path("unknown_person_1.jpg")]}
        audit = {"excluded_identities": [], "reserve_replacements": [], "preflight_records": []}
        if gallery_side_effect is None:
            gallery_patch = patch.object(
                evaluate_open_set,
                "build_gallery",
                return_value={"names": ["known_person"], "embeddings": np.ones((1, 2)),
                              "errors": [], "pending": []},
            )
        else:
            gallery_patch = patch.object(
                evaluate_open_set, "build_gallery", side_effect=gallery_side_effect
            )
        patches = [
            patch.object(evaluate_open_set, "OUTPUT_ROOT", self.output_root),
            patch.object(evaluate_open_set, "DATA_ROOT", self.root / "data"),
            patch.object(evaluate_open_set, "discover_eligible_identities", return_value={}),
            patch.object(evaluate_open_set, "FaceDetector", return_value=Mock()),
            patch.object(evaluate_open_set, "FaceEmbedder", return_value=Mock()),
            patch.object(evaluate_open_set, "make_split", return_value=(known, unknown, audit)),
            gallery_patch,
            patch.object(evaluate_open_set.shutil, "copy2", side_effect=lambda source, target: Path(target).write_bytes(b"fake")),
            patch.object(evaluate_open_set, "classify_query", return_value={
                "status": "evaluated", "identity": "known_person",
                "match_status": "recognised", "best_score": 0.9,
                "second_score": 0.1, "margin": 0.8,
            }),
        ]
        return patches

    def test_evaluation_failure_cleans_temporary_output(self):
        patches = self._patch_run_dependencies(gallery_side_effect=RuntimeError("simulated failure"))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                evaluate(self.image_root, self.final_dir, known_count=1, unknown_count=1)

        self.assertFalse(self.final_dir.exists())
        self.assertEqual(list(self.output_root.iterdir()), [])

    def test_existing_output_directory_is_never_overwritten(self):
        self.final_dir.mkdir()
        sentinel = self.final_dir / "sentinel.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        patches = self._patch_run_dependencies()
        with patches[0], patches[1]:
            with self.assertRaises(FileExistsError):
                evaluate(self.image_root, self.final_dir, known_count=1, unknown_count=1)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(list(self.output_root.iterdir()), [self.final_dir])

    def test_final_output_appears_only_after_both_reports_are_written(self):
        real_writer = evaluate_open_set.write_manifest_exclusive
        written = []

        def inspect_write(path, manifest):
            self.assertFalse(self.final_dir.exists())
            result = real_writer(path, manifest)
            written.append(Path(path).name)
            return result

        patches = self._patch_run_dependencies()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], \
                patch.object(evaluate_open_set, "write_manifest_exclusive", side_effect=inspect_write):
            evaluate(self.image_root, self.final_dir, known_count=1, unknown_count=1)

        self.assertEqual(written, ["open_set_report.json", "reproducibility_manifest.json"])
        self.assertTrue((self.final_dir / "open_set_report.json").is_file())
        self.assertTrue((self.final_dir / "reproducibility_manifest.json").is_file())
        self.assertEqual(list(self.output_root.iterdir()), [self.final_dir])


if __name__ == "__main__":
    unittest.main()
