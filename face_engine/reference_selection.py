"""Experimental selection policy for reference enrollment."""

from math import isfinite
from typing import Literal


SMART = "smart"
ALWAYS_REVIEW = "always_review"
ReferenceSelectionPolicy = Literal["smart", "always_review"]
MIN_AREA_RATIO = 5.0
MIN_DETECTION_SCORE = 0.75


def valid_reference_face(face):
    x, y, width, height = map(float, face["bbox"])
    area = width * height
    return (width > 0 and height > 0 and area > 0
            and all(isfinite(value) for value in (x, y, area, face["confidence"])))


def dominant_face_selection(faces, policy=SMART):
    """Return selection evidence with the original candidate index, or None."""
    if policy != SMART or len(faces) < 2:
        return None

    areas = []
    scores = []
    for face in faces:
        if not valid_reference_face(face):
            return None
        _, _, width, height = face["bbox"]
        areas.append(float(width) * float(height))
        scores.append(float(face["confidence"]))

    largest, second = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)[:2]
    ratio = areas[largest] / areas[second]
    score = scores[largest]
    if ratio < MIN_AREA_RATIO or score < MIN_DETECTION_SCORE:
        return None
    if any(other >= score for index, other in enumerate(scores) if index != largest):
        return None
    return {
        "reason": "dominant_face",
        "selected_face_index": largest,
        "area_ratio": ratio,
        "detection_score": score,
    }
