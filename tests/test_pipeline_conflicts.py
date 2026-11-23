"""Synthetic regression fixtures for image-level duplicate identity handling.

These scores are constructed test inputs; they are not recovered historical
Tagore/Einstein predictions.
"""

import unittest

import numpy as np

from face_engine.pipeline import recognize_image


class StubDetector:
    def __init__(self, count):
        self.count = count

    def detect(self, image):
        return [
            {"bbox": [index, 0, 1, 1], "confidence": 0.99}
            for index in range(self.count)
        ]


class StubEmbedder:
    def get_embedding(self, image, detection):
        return detection["bbox"][0]


class StubMatcher:
    possible_threshold = 0.35

    def __init__(self, predictions):
        self.predictions = predictions

    def match(self, embedding, gallery):
        return dict(self.predictions[embedding])


def prediction(identity, status, score):
    return {
        "identity": identity,
        "status": status,
        "best_score": score,
        "second_score": max(-1.0, score - 0.2),
        "margin": 0.2,
        "candidates": [{"name": identity, "score": score}] if identity else [],
    }


def run_pipeline(predictions):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return recognize_image(
        image,
        {"names": ["GalleryIdentity"]},
        StubDetector(len(predictions)),
        StubEmbedder(),
        StubMatcher(predictions),
    )["results"]


class DuplicateIdentityConflictTests(unittest.TestCase):
    def test_higher_scoring_duplicate_is_kept_and_lower_becomes_possible(self):
        results = run_pipeline({
            0: prediction("person-a", "recognised", 0.81),
            1: prediction("person-a", "recognised", 0.62),
        })
        self.assertEqual(results[0]["status"], "recognised")
        self.assertEqual(results[0]["best_score"], 0.81)
        self.assertNotIn("conflict_with", results[0])
        self.assertEqual(results[1]["status"], "possible")
        self.assertEqual(results[1]["best_score"], 0.62)
        self.assertEqual(results[1]["conflict_with"], 1)

    def test_conflicting_score_below_possible_threshold_becomes_unknown(self):
        results = run_pipeline({
            0: prediction("person-a", "recognised", 0.81),
            1: prediction("person-a", "possible", 0.34),
        })
        self.assertEqual(results[1]["status"], "unknown")
        self.assertEqual(results[1]["identity"], "person-a")
        self.assertEqual(results[1]["best_score"], 0.34)
        self.assertEqual(results[1]["conflict_with"], 1)

    def test_exact_possible_threshold_is_inclusive(self):
        results = run_pipeline({
            0: prediction("person-a", "recognised", 0.8),
            1: prediction("person-a", "possible", 0.35),
        })
        self.assertEqual(results[1]["status"], "possible")

    def test_three_conflicts_all_reference_the_highest_scoring_face(self):
        results = run_pipeline({
            0: prediction("person-a", "recognised", 0.88),
            1: prediction("person-a", "possible", 0.45),
            2: prediction("person-a", "recognised", 0.70),
            3: prediction("person-a", "possible", 0.20),
        })
        self.assertEqual([item["status"] for item in results], [
            "recognised", "possible", "possible", "unknown",
        ])
        self.assertEqual([item.get("conflict_with") for item in results], [
            None, 1, 1, 1,
        ])

    def test_equal_scores_keep_the_earlier_detected_face(self):
        results = run_pipeline({
            0: prediction("person-a", "possible", 0.6),
            1: prediction("person-a", "recognised", 0.6),
        })
        self.assertEqual(results[0]["status"], "possible")
        self.assertNotIn("conflict_with", results[0])
        self.assertEqual(results[1]["status"], "possible")
        self.assertEqual(results[1]["conflict_with"], 1)

    def test_different_identities_are_unaffected(self):
        results = run_pipeline({
            0: prediction("person-a", "recognised", 0.8),
            1: prediction("person-b", "recognised", 0.7),
        })
        self.assertEqual([item["status"] for item in results], ["recognised", "recognised"])
        self.assertTrue(all("conflict_with" not in item for item in results))

    def test_unknown_without_identity_is_unaffected(self):
        results = run_pipeline({
            0: prediction("person-a", "recognised", 0.8),
            1: prediction(None, "unknown", 0.3),
        })
        self.assertEqual(results[1]["status"], "unknown")
        self.assertIsNone(results[1]["identity"])
        self.assertNotIn("conflict_with", results[1])


if __name__ == "__main__":
    unittest.main()
