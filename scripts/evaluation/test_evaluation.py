"""Focused unit tests for WIDER FACE evaluation selection and safety."""

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation import evaluate_widerface as widerface
from evaluation.common import validate_output_path


class FakeDetector:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def detect(self, image):
        return []


class WiderFaceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.permitted_evaluation_root = (
            Path(__file__).resolve().parents[2]
            / "data/private/evaluation"
        )
        self.permitted_evaluation_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix=".test-evaluation-",
            dir=self.permitted_evaluation_root,
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assertEqual(self.root.parent.resolve(), self.permitted_evaluation_root.resolve())
        self.dataset = self.root / "dataset"
        self.image_root = self.dataset / "images"
        self.output_root = self.root / "evaluation"
        self.image_root.mkdir(parents=True)
        self.output_root.mkdir()
        self.entries = [
            ("one.jpg", [[0, 0, 10, 10]]),
            ("many.jpg", [[0, 0, 10, 10], [20, 20, 30, 30]]),
            ("also-one.jpg", [[1, 1, 11, 11]]),
            ("zero.jpg", []),
        ]
        self.patches = [
            patch.object(widerface, "ANNOTATION_FILE", self.root / "annotations.txt"),
            patch.object(widerface, "IMAGE_ROOT", self.image_root),
            patch.object(widerface, "DATASET_ROOT", self.dataset),
            patch.object(widerface, "EVALUATION_OUTPUT_DIR", self.output_root),
            patch.object(widerface, "load_annotations", return_value=self.entries),
            patch.object(widerface, "FaceDetector", FakeDetector),
            patch.object(widerface.cv2, "imread", return_value=object()),
        ]
        for item in self.patches:
            item.start()
        (self.root / "annotations.txt").write_text("fixture", encoding="utf-8")

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def run_eval(self, name, **kwargs):
        output = self.output_root / name
        widerface.evaluate(
            manifest_path=output,
            **kwargs,
        )
        return json.loads(output.read_text(encoding="utf-8"))

    def test_mixed_sampling_and_manifest_remain_backward_compatible(self):
        manifest = self.run_eval("mixed.json", limit=2, seed=42)
        shuffled = list(self.entries)
        random.Random(42).shuffle(shuffled)
        self.assertEqual(
            [item["image"] for item in manifest["sampled_images"]],
            [path for path, _ in shuffled[:2]],
        )
        self.assertEqual(manifest["requested_limit"], 2)
        self.assertEqual(manifest["seed"], 42)
        self.assertNotIn("filter", manifest)
        self.assertNotIn("eligible_image_count", manifest)
        self.assertEqual(
            set(manifest),
            {
                "evaluation_type", "official_benchmark_ap", "method_note",
                "dataset", "annotation_file", "image_root", "seed",
                "requested_limit", "sampled_image_count", "iou_threshold",
                "model_name", "detector_parameters", "counts", "custom_metrics",
                "denominator_policy", "sampled_images", "excluded_samples",
            },
        )

    def test_legacy_cli_defaults_are_unchanged(self):
        with patch.object(widerface, "evaluate") as evaluate_call:
            widerface.main([])
        kwargs = evaluate_call.call_args.kwargs
        self.assertEqual(kwargs["limit"], 100)
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["iou_threshold"], 0.5)
        self.assertEqual(kwargs["score_threshold"], 0.5)
        self.assertEqual(kwargs["nms_threshold"], 0.3)
        self.assertEqual(kwargs["max_dimension"], 1280)

    def test_exact_face_filter_precedes_seeded_sampling_and_records_metadata(self):
        manifest = self.run_eval(
            "single.json", limit=1, seed=42, exact_faces=1
        )
        eligible = [entry for entry in self.entries if len(entry[1]) == 1]
        random.Random(42).shuffle(eligible)
        self.assertEqual(manifest["eligible_image_count"], 2)
        self.assertEqual(manifest["filter"], {"exact_ground_truth_faces": 1})
        self.assertEqual(manifest["selected_image_paths"], [eligible[0][0]])
        self.assertEqual(manifest["sampled_images"][0]["image"], eligible[0][0])
        self.assertEqual(manifest["seed"], 42)

    def test_exact_face_filter_defaults_to_all_eligible_images(self):
        manifest = self.run_eval("all-single.json", exact_faces=1)
        self.assertEqual(manifest["eligible_image_count"], 2)
        self.assertEqual(manifest["sampled_image_count"], 2)
        self.assertEqual(set(manifest["selected_image_paths"]), {"one.jpg", "also-one.jpg"})

    def test_empty_eligible_subset_fails_before_detector_or_output(self):
        with patch.object(widerface, "FaceDetector") as detector:
            output = self.output_root / "empty.json"
            with self.assertRaisesRegex(ValueError, "No WIDER FACE images"):
                widerface.evaluate(
                    exact_faces=99,
                    manifest_path=output,
                )
            detector.assert_not_called()
            self.assertFalse(output.exists())

    def test_output_validation_rejects_protected_and_existing_paths(self):
        with self.assertRaises(ValueError):
            validate_output_path(self.dataset / "manifest.json", self.output_root, (self.dataset,))
        existing = self.output_root / "existing.json"
        existing.write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            validate_output_path(existing, self.output_root, (self.dataset,))

    def test_output_validation_rejects_hardlinks(self):
        linked = self.output_root / "linked.json"
        linked.write_text("{}", encoding="utf-8")
        second_link = self.output_root / "linked-copy.json"
        try:
            os.link(linked, second_link)
        except OSError as error:
            self.skipTest(f"Hard links are unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "multiple hard links"):
            validate_output_path(linked, self.output_root, (self.dataset,))

    def test_output_validation_rejects_symlinked_output_roots(self):
        symlink = self.root / "evaluation-link"
        try:
            symlink.symlink_to(self.output_root, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks are unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate_output_path(symlink / "new.json", symlink, (self.dataset,))


if __name__ == "__main__":
    unittest.main()
