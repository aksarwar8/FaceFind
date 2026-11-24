"""Focused tests for targeted WIDER FACE diagnostic utilities."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation import diagnose_widerface_errors as diagnostics
from evaluation.widerface_common import EVALUATION_OUTPUT_DIR


class DiagnosticUtilityTests(unittest.TestCase):
    def setUp(self):
        self.eval_root = Path(EVALUATION_OUTPUT_DIR).resolve()
        self.eval_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix=".test-widerface-diagnostics-",
            dir=self.eval_root,
        )
        self.addCleanup(self.temp.cleanup)
        self.temp_root = Path(self.temp.name)

    def test_iou_and_confidence_ordered_duplicate_matching(self):
        self.assertEqual(diagnostics.calculate_iou([0, 0, 10, 10], [0, 0, 20, 20]), 0.25)
        result = diagnostics.match_predictions(
            [
                {"bbox": [10, 10, 20, 20], "confidence": 0.95},
                {"bbox": [10, 10, 20, 20], "confidence": 0.80},
                {"bbox": [40, 40, 10, 10], "confidence": 0.50},
            ],
            [[10, 10, 30, 30]],
            iou_threshold=0.5,
        )
        self.assertEqual(
            result["counts"],
            {"true_positives": 1, "false_positives": 2, "false_negatives": 0},
        )
        first, duplicate, unrelated = result["predicted_boxes"]
        self.assertTrue(first["matches_iou_threshold"])
        self.assertEqual(first["best_iou"], 1.0)
        self.assertFalse(duplicate["matches_iou_threshold"])
        self.assertEqual(duplicate["best_iou"], 1.0)
        self.assertEqual(duplicate["available_best_iou"], 0.0)
        self.assertEqual(unrelated["best_iou"], 0.0)

    def test_prediction_below_iou_threshold_becomes_fp_and_gt_fn(self):
        result = diagnostics.match_predictions(
            [{"bbox": [0, 0, 10, 10], "confidence": 0.9}],
            [[0, 0, 20, 20]],
            iou_threshold=0.5,
        )
        self.assertEqual(result["predicted_boxes"][0]["best_iou"], 0.25)
        self.assertEqual(
            result["counts"],
            {"true_positives": 0, "false_positives": 1, "false_negatives": 1},
        )

    def test_diagnostic_record_contains_geometry_confidence_and_counts(self):
        record = diagnostics.build_diagnostic_record(
            "sample/image.jpg",
            {
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 0,
                "predicted_faces": 1,
            },
            [[5, 6, 25, 26]],
            [{"bbox": [5, 6, 20, 20], "confidence": 0.91}],
            0.5,
            (100, 120, 3),
        )
        self.assertEqual(record["image_dimensions"], {"height": 100, "width": 120})
        self.assertEqual(record["ground_truth_boxes"][0]["bbox_xyxy"], [5.0, 6.0, 25.0, 26.0])
        self.assertEqual(record["predicted_boxes"][0]["confidence"], 0.91)
        self.assertEqual(record["counts"]["true_positives"], 1)

    def test_case_selection_is_seeded_and_tracks_fn_fp_categories(self):
        rows = [
            {
                "image": f"image-{index}.jpg",
                "status": "evaluated",
                "false_negatives": int(index < 3),
                "false_positives": int(index >= 2),
            }
            for index in range(6)
        ]
        first, selection = diagnostics.select_error_cases(
            {"seed": 42, "sampled_images": rows}, num_fn=2, num_fp=2
        )
        second, _ = diagnostics.select_error_cases(
            {"seed": 42, "sampled_images": rows}, num_fn=2, num_fp=2
        )
        self.assertEqual(list(first), list(second))
        self.assertEqual(selection["selected_fn_cases"], 2)
        self.assertEqual(selection["selected_fp_cases"], 2)
        self.assertEqual(selection["available_fn_cases"], 3)
        self.assertEqual(selection["available_fp_cases"], 4)

    def test_output_directory_must_be_new_direct_child_under_evaluation_root(self):
        candidate = self.eval_root / f"{self.temp_root.name}-fresh-run"
        self.assertEqual(
            diagnostics.validate_diagnostic_output_dir(candidate),
            candidate.resolve(),
        )
        candidate.mkdir()
        self.addCleanup(candidate.rmdir)
        with self.assertRaises(FileExistsError):
            diagnostics.validate_diagnostic_output_dir(candidate)
        with self.assertRaises(ValueError):
            diagnostics.validate_diagnostic_output_dir(self.eval_root.parent / "outside-run")

    def test_symlinked_evaluation_root_is_rejected(self):
        alias = self.temp_root.parent / "evaluation-root-link"
        try:
            alias.symlink_to(self.temp_root, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks are unavailable: {error}")
        self.addCleanup(alias.unlink)
        with patch.object(diagnostics, "EVALUATION_OUTPUT_DIR", alias):
            with self.assertRaisesRegex(ValueError, "symlink"):
                diagnostics.validate_diagnostic_output_dir(alias / "new-run")


if __name__ == "__main__":
    unittest.main()
