
from pathlib import Path

import cv2


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face"
    / "face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    def __init__(self, score_threshold=0.7, nms_threshold=0.3):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YuNet model not found: {MODEL_PATH}")

        self.detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            5000,
        )

    def detect(self, image):
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        results = []

        if faces is None:
            return results

        for face in faces:
            results.append({
                "bbox": face[:4].astype(int).tolist(),
                "landmarks": face[4:14].tolist(),
                "confidence": float(face[14]),
            })

        return results
