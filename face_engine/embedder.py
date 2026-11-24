
from pathlib import Path

import cv2
import numpy as np
import time


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face"
    / "face_recognition_sface_2021dec.onnx"
)


class FaceEmbedder:
    dimension = 128

    def __init__(self):
        if not MODEL_PATH.is_file():
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
        align_started = time.perf_counter()
        aligned_face = self.recognizer.alignCrop(
            image,
            face_row,
        )
        alignment_ms = (time.perf_counter() - align_started) * 1000

        # Extract the 128-dimensional feature vector.
        embedding_started = time.perf_counter()
        embedding = self.recognizer.feature(aligned_face)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        # Normalize the embedding.
        embedding = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Invalid zero-length face embedding")

        self.last_timings_ms = {"alignment": alignment_ms, "embedding": embedding_ms}
        return embedding / norm
