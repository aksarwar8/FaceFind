"""Focused tests for the fixed-manifest WIDER FACE detector comparison."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts.evaluation import compare_widerface_detectors as comparison


class MatchingAndLatencyTests(unittest.TestCase):
    def test_greedy_matching_is_one_to_one_and_reports_height_recall(self):
        ground_truth = [
            [0, 0, 10, 10],
            [20, 0, 40, 30],
            [50, 0, 100, 60],
        ]
        predictions = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.99},
            {"bbox": [0, 0, 10, 10], "confidence": 0.90},
            {"bbox": [20, 0, 40, 30], "confidence": 0.80},
            {"bbox": [50, 0, 100, 60], "confidence": 0.70},
        ]

        result = comparison.match_image_predictions(ground_truth, predictions)

        self.assertEqual(result["true_positives"], 3)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(result["matched_ground_truth_by_height"], {
            "le_20_px": 1, "gt_20_to_40_px": 1, "gt_40_px": 1,
        })
        self.assertEqual(result["ground_truth_by_height"], {
            "le_20_px": 1, "gt_20_to_40_px": 1, "gt_40_px": 1,
        })

    def test_latency_summary_reports_mean_median_and_p95(self):
        result = comparison._latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["p95_ms"], 3.85)

    def test_resize_preserves_aspect_and_only_downscales(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized, scale = comparison.resize_preserving_aspect(image, 100)
        self.assertEqual(resized.shape[:2], (64, 128))
        self.assertEqual(scale, 0.5)
        unchanged, unchanged_scale = comparison.resize_preserving_aspect(image, 300)
        self.assertEqual(unchanged.shape[:2], (128, 224))
        self.assertTrue(np.array_equal(unchanged[:100, :200], image))
        self.assertEqual(unchanged_scale, 1.0)


class SCRFDAdapterTests(unittest.TestCase):
    def test_maps_box_score_and_five_landmarks(self):
        model = Mock()
        model.use_kps = True
        model.detect.return_value = (
            np.asarray([[1, 2, 11, 22, 0.9]], dtype=np.float32),
            np.asarray([[[2, 5], [8, 5], [5, 10], [3, 17], [7, 17]]], dtype=np.float32),
        )
        result = comparison.SCRFDKPSAdapter(model).detect(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(result[0]["bbox"], [1.0, 2.0, 11.0, 22.0])
        self.assertAlmostEqual(result[0]["confidence"], 0.9, places=6)
        self.assertEqual(len(result[0]["landmarks"]), 10)
        self.assertEqual(model.detect.call_args.kwargs["input_size"], (32, 32))

    def test_rejects_non_kps_checkpoint_outputs(self):
        model = Mock()
        model.use_kps = True
        model.detect.return_value = (np.asarray([[1, 2, 11, 22, 0.9]]), None)
        with self.assertRaisesRegex(RuntimeError, "five-landmark KPS variant"):
            comparison.SCRFDKPSAdapter(model).detect(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_rejects_checkpoint_that_does_not_advertise_kps(self):
        model = Mock()
        model.use_kps = False
        with self.assertRaisesRegex(ValueError, "five-point KPS outputs"):
            comparison.SCRFDKPSAdapter(model)


class FailureCategoryTests(unittest.TestCase):
    def _result(self):
        return {"counts": comparison._empty_counts(), "latency_samples_ms": []}

    def _assert_failure_category(self, category, image, adapter, resize_patch=None,
                                 matching_patch=None):
        result = self._result()
        gt_boxes = [[0, 0, 10, 10]]
        with patch.object(comparison, "resize_preserving_aspect", resize_patch or
                          comparison.resize_preserving_aspect), \
                patch.object(comparison, "match_image_predictions", matching_patch or
                             comparison.match_image_predictions):
            row = comparison._evaluate_detector_image(
                result, image, "event/image.jpg", gt_boxes, adapter, 1280
            )
        self.assertEqual(row["reason"], category)
        self.assertEqual(result["counts"]["failure_categories"][category], 1)
        self.assertEqual(result["counts"]["false_negatives"], 1)
        self.assertEqual(result["counts"]["images_evaluated"], 0)
        return row

    def test_read_none_is_reported_as_image_read_failure(self):
        with patch.object(comparison.cv2, "imread", return_value=None):
            image, error = comparison._read_evaluation_image(Path("missing.jpg"))
        self.assertIsNone(image)
        self.assertEqual(error, "cv2.imread returned None")
        result = self._result()
        row = {"image": "event/missing.jpg", "ground_truth_faces": 2}
        comparison._record_image_read_failure(result, row, error, [[0, 0, 1, 1], [2, 2, 3, 3]])
        self.assertEqual(row["reason"], "image_read_failure")
        self.assertEqual(row["error"], "cv2.imread returned None")
        self.assertEqual(result["counts"]["failure_categories"]["image_read_failure"], 1)
        self.assertEqual(result["counts"]["unreadable_ground_truth_faces_excluded"], 2)
        self.assertEqual(result["counts"]["false_negatives"], 0)

    def test_read_exception_is_reported_with_exception_type_and_message(self):
        with patch.object(comparison.cv2, "imread", side_effect=OSError("decoder failed")):
            image, error = comparison._read_evaluation_image(Path("broken.jpg"))
        self.assertIsNone(image)
        self.assertEqual(error, "OSError: decoder failed")

    def test_preprocessing_exception_has_its_own_category(self):
        self._assert_failure_category(
            "preprocessing_exception", np.zeros((4, 4, 3), dtype=np.uint8), Mock(),
            resize_patch=Mock(side_effect=ValueError("resize failed")),
        )

    def test_detector_exception_has_its_own_category(self):
        adapter = Mock()
        adapter.detect.side_effect = RuntimeError("inference failed")
        self._assert_failure_category(
            "detector_exception", np.zeros((4, 4, 3), dtype=np.uint8), adapter
        )

    def test_prediction_validation_exception_has_its_own_category(self):
        adapter = Mock()
        adapter.detect.return_value = [{"bbox": [1, 2], "confidence": 0.9}]
        self._assert_failure_category(
            "prediction_validation_exception", np.zeros((4, 4, 3), dtype=np.uint8), adapter
        )

    def test_matching_exception_has_its_own_category(self):
        adapter = Mock()
        adapter.detect.return_value = []
        self._assert_failure_category(
            "matching_exception", np.zeros((4, 4, 3), dtype=np.uint8), adapter,
            matching_patch=Mock(side_effect=RuntimeError("matching failed")),
        )


class FixedManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=comparison.EVALUATION_OUTPUT_DIR)
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "manifest.json"
        self.image_root = self.root / "images"
        self.image_root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def _write_manifest(self, records):
        value = {
            "image_root": str(self.image_root),
            "annotation_file": str(self.root / "annotations.txt"),
            "sampled_image_count": 500,
            "sampled_images": records,
        }
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")

    def test_selects_manifest_prefix_without_resampling(self):
        records = [
            {"image": f"event/{index:03}.jpg", "status": "evaluated", "ground_truth_faces": 1}
            for index in range(500)
        ]
        self._write_manifest(records)
        with patch.object(comparison, "EVALUATION_OUTPUT_DIR", self.root), \
                patch.object(comparison, "IMAGE_ROOT", self.image_root), \
                patch.object(comparison, "ANNOTATION_FILE", self.root / "annotations.txt"), \
                patch.object(comparison, "validate_image_path"):
            _, _, selected = comparison.load_fixed_manifest(self.manifest_path, 25)

        self.assertEqual(len(selected), 25)
        self.assertEqual([path for path, _ in selected], [f"event/{i:03}.jpg" for i in range(25)])

    def test_rejects_duplicate_manifest_image_paths(self):
        records = [
            {"image": f"event/{index:03}.jpg", "status": "evaluated", "ground_truth_faces": 1}
            for index in range(500)
        ]
        records[10]["image"] = records[9]["image"]
        self._write_manifest(records)
        with patch.object(comparison, "EVALUATION_OUTPUT_DIR", self.root), \
                patch.object(comparison, "IMAGE_ROOT", self.image_root), \
                patch.object(comparison, "ANNOTATION_FILE", self.root / "annotations.txt"), \
                patch.object(comparison, "validate_image_path"):
            with self.assertRaisesRegex(ValueError, "Duplicate image path"):
                comparison.load_fixed_manifest(self.manifest_path, 25)

    def test_requires_matching_annotation_path_and_face_count(self):
        selected = [("event/001.jpg", {"ground_truth_faces": 2})]
        comparison.validate_annotation_matches(selected, {"event/001.jpg": [[0, 0, 10, 10], [20, 0, 30, 10]]})
        with self.assertRaisesRegex(ValueError, "face count mismatch"):
            comparison.validate_annotation_matches(selected, {"event/001.jpg": [[0, 0, 10, 10]]})
        with self.assertRaisesRegex(ValueError, "no annotation entry"):
            comparison.validate_annotation_matches(selected, {})

    def test_existing_output_directory_is_refused_before_model_setup(self):
        output_root = self.root / "evaluation"
        output_root.mkdir()
        output_dir = output_root / "existing-run"
        output_dir.mkdir()
        sentinel = output_dir / "keep.txt"
        sentinel.write_text("existing", encoding="utf-8")
        with patch.object(comparison, "EVALUATION_OUTPUT_DIR", output_root), \
                patch.object(comparison, "DATASET_ROOT", self.root / "dataset"), \
                patch.object(comparison, "ANNOTATION_FILE", self.root / "annotations.txt"):
            with self.assertRaises(FileExistsError):
                comparison.run_comparison(self.manifest_path, output_dir, self.root / "model.onnx")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
