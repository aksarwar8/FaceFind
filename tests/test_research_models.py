import unittest
from unittest.mock import Mock

import numpy as np

from face_engine.research_models import (
    ADAFACE_MARGIN_THRESHOLD,
    ADAFACE_SIMILARITY_THRESHOLD,
    AdaFaceGalleryMatcher,
    StridePaddedSCRFD,
)


class FakeSCRFDAdapter:
    def detect(self, image):
        self.shape = image.shape
        return [{
            "bbox": [10.0, 20.0, 60.0, 80.0],
            "landmarks": [20, 35, 45, 35, 33, 48, 23, 62, 42, 62],
            "confidence": 0.9,
        }]


class ResearchModelTests(unittest.TestCase):
    def test_reference_keeps_primary_matches_without_retry(self):
        detector = StridePaddedSCRFD.__new__(StridePaddedSCRFD)
        detector.adapter = Mock(wraps=FakeSCRFDAdapter())
        image = np.zeros((800, 1280, 3), dtype=np.uint8)
        faces, trace = detector.detect_reference(image)
        detector.adapter.detect.assert_called_once()
        self.assertEqual(faces[0]["bbox"], [10.0, 20.0, 50.0, 60.0])
        self.assertEqual(trace["pass"], "primary")
        self.assertIsNone(trace["retry_ms"])

    def test_reference_retry_pads_and_maps_ordered_landmarks_to_working_image(self):
        detector = StridePaddedSCRFD.__new__(StridePaddedSCRFD)
        retry = FakeSCRFDAdapter().detect(np.zeros((1, 1, 3), dtype=np.uint8))
        # A primary box entirely in padding is clipped to zero area.
        primary = [{**retry[0], "bbox": [0, 768, 20, 800]}]
        detector.adapter = Mock()
        detector.adapter.detect.side_effect = [primary, retry]
        image = np.zeros((750, 1280, 3), dtype=np.uint8)
        faces, trace = detector.detect_reference(image)
        calls = detector.adapter.detect.call_args_list
        self.assertEqual([call.args[0].shape[:2] for call in calls], [(768, 1280), (384, 640)])
        self.assertTrue(np.all(calls[1].args[0][375:] == 0))
        self.assertEqual(faces[0]["bbox"], [20.0, 40.0, 100.0, 120.0])
        self.assertEqual(faces[0]["landmarks"], [value * 2 for value in retry[0]["landmarks"]])
        self.assertEqual(trace["pass"], "retry_640")
        self.assertGreaterEqual(trace["retry_ms"], 0.0)

    def test_scrfd_input_is_stride_padded_and_coordinates_are_xywh(self):
        detector = StridePaddedSCRFD.__new__(StridePaddedSCRFD)
        detector.adapter = FakeSCRFDAdapter()
        detector.stride = 32
        image = np.zeros((250, 250, 3), dtype=np.uint8)
        rows = detector.detect(image)
        self.assertEqual(detector.adapter.shape, (256, 256, 3))
        self.assertEqual(rows[0]["bbox"], [10.0, 20.0, 50.0, 60.0])
        self.assertEqual(len(rows[0]["landmarks"]), 10)

    def test_adaface_uses_calibrated_open_set_threshold_and_margin(self):
        matcher = AdaFaceGalleryMatcher()
        vectors = np.eye(512, dtype=np.float32)[:50]
        gallery = {"names": [f"person_{i}" for i in range(50)], "embeddings": vectors}
        accepted = matcher.match(vectors[0], gallery)
        self.assertEqual(accepted["status"], "recognised")
        self.assertEqual(accepted["identity"], "person_0")
        self.assertEqual(matcher.recognition_threshold, ADAFACE_SIMILARITY_THRESHOLD)
        self.assertEqual(matcher.margin_threshold, ADAFACE_MARGIN_THRESHOLD)
        rejected = matcher.match(np.ones(512, dtype=np.float32) / np.sqrt(512), gallery)
        self.assertEqual(rejected["status"], "unknown")

    def test_duplicate_adaface_assignment_becomes_unknown(self):
        self.assertEqual(AdaFaceGalleryMatcher.duplicate_status({}), "unknown")


if __name__ == "__main__":
    unittest.main()
