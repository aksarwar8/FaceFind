
from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face"
    / "face_recognition_sface_2021dec.onnx"
)


class FaceEmbedder:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"SFace model not found: {MODEL_PATH}"
            )

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(MODEL_PATH),
            "",
        )

    def get_embedding(self, image, detection):
        # Reconstruct YuNet's original 15-value face output.
        face_row = np.array(
            [
                *detection["bbox"],
                *detection["landmarks"],
                detection["confidence"],
            ],
            dtype=np.float32,
        )

        # Align the face using its five facial landmarks.
        aligned_face = self.recognizer.alignCrop(
            image,
            face_row,
        )

        # Extract the 128-dimensional feature vector.
        embedding = self.recognizer.feature(aligned_face)

        # Normalize the embedding.
        embedding = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Invalid zero-length face embedding")

        return embedding / norm
