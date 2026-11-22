
from pathlib import Path

import cv2


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face"
    / "face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    def __init__(
        self,
        score_threshold=0.5,
        nms_threshold=0.3,
        max_dimension=1280,
    ):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {MODEL_PATH}"
            )

        self.max_dimension = max_dimension

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

        original_height, original_width = image.shape[:2]

        # Resize large images without changing aspect ratio.
        scale = min(
            self.max_dimension / max(original_width, original_height),
            1.0,
        )

        if scale < 1.0:
            resized_width = int(original_width * scale)
            resized_height = int(original_height * scale)

            detection_image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = image

        resized_height, resized_width = detection_image.shape[:2]

        self.detector.setInputSize(
            (resized_width, resized_height)
        )

        _, faces = self.detector.detect(detection_image)

        results = []

        if faces is None:
            return results

        # Convert detections back to original-image coordinates.
        for face in faces:
            bbox = face[:4].copy() / scale
            landmarks = face[4:14].copy() / scale

            results.append({
                "bbox": bbox.astype(int).tolist(),
                "landmarks": landmarks.tolist(),
                "confidence": float(face[14]),
            })

        return results
