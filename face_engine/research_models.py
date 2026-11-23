"""Offline local demo adapters for SCRFD KPS and AdaFace IR-18.

This is an experimental application configuration. YuNet/SFace remains the
fallback implementation and its thresholds are unchanged.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform


ROOT = Path(__file__).resolve().parents[1]
SCRFD_MODEL = ROOT / "data/private/models/scrfd_10g_bnkps.onnx"
ADAFACE_MODEL = ROOT / "data/private/models/adaface_ir_18_webface4m_yakhyo.onnx"
SCRFD_SCORE_THRESHOLD = 0.5
SCRFD_NMS_THRESHOLD = 0.4
SCRFD_MAX_DIMENSION = 1280
# Independently calibrated on 50-way group-photo gallery identities in the
# disjoint end-to-end calibration cohort.
ADAFACE_SIMILARITY_THRESHOLD = 0.26362
ADAFACE_MARGIN_THRESHOLD = 0.07940
ALIGN_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                      [56.0252, 71.7366], [41.5493, 92.3655],
                      [70.7299, 92.2041]], dtype=np.float32)


class StridePaddedSCRFD:
    """Reuse the evaluated SCRFD KPS adapter with stride-32 input padding."""
    def __init__(self):
        import sys
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from evaluation.compare_widerface_detectors import create_scrfd_adapter
        self.adapter = create_scrfd_adapter(
            SCRFD_MODEL,
            score_threshold=SCRFD_SCORE_THRESHOLD,
            nms_threshold=SCRFD_NMS_THRESHOLD,
            max_dimension=SCRFD_MAX_DIMENSION,
        )

    def detect(self, image):
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        original_height, original_width = image.shape[:2]
        scale = min(SCRFD_MAX_DIMENSION / max(original_width, original_height), 1.0)
        if scale < 1.0:
            width, height = int(original_width * scale), int(original_height * scale)
            work = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        else:
            work = image
        height, width = work.shape[:2]
        padded_height = int(math.ceil(height / 32) * 32)
        padded_width = int(math.ceil(width / 32) * 32)
        if (padded_height, padded_width) != (height, width):
            padded = np.zeros((padded_height, padded_width, 3), dtype=work.dtype)
            padded[:height, :width] = work
            work = padded
        rows = self.adapter.detect(work)
        results = []
        for row in rows:
            x1, y1, x2, y2 = map(float, row["bbox"])
            points = np.asarray(row["landmarks"], dtype=np.float32).reshape(5, 2)
            x1, x2 = np.clip(np.array([x1, x2]) / scale, 0, original_width)
            y1, y2 = np.clip(np.array([y1, y2]) / scale, 0, original_height)
            points /= scale
            results.append({
                "bbox": [float(x1), float(y1), float(max(0, x2 - x1)), float(max(0, y2 - y1))],
                "landmarks": points.reshape(-1).astype(float).tolist(),
                "confidence": float(row["confidence"]),
            })
        return results


class AdaFaceEmbedder:
    dimension = 512
    cache_key = "scrfd_adaface_v1"

    def __init__(self):
        if not ADAFACE_MODEL.is_file():
            raise FileNotFoundError(f"AdaFace checkpoint not found: {ADAFACE_MODEL}")
        self.session = ort.InferenceSession(str(ADAFACE_MODEL), providers=["CPUExecutionProvider"])
        if self.session.get_providers()[0] != "CPUExecutionProvider":
            raise RuntimeError("AdaFace CPU execution provider was not selected")
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        if len(self.input.shape) != 4 or self.input.shape[1:] != [3, 112, 112]:
            raise ValueError(f"Unsupported AdaFace input contract: {self.input.shape}")
        if len(self.output.shape) != 2 or self.output.shape[1] != self.dimension:
            raise ValueError(f"Unsupported AdaFace output contract: {self.output.shape}")

    def get_embedding(self, image, detection):
        started = time.perf_counter()
        points = np.asarray(detection["landmarks"], dtype=np.float32).reshape(5, 2)
        transform = SimilarityTransform()
        if not transform.estimate(points, ALIGN_DST):
            raise ValueError("AdaFace five-point alignment failed")
        matrix = transform.params[:2, :].astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("AdaFace alignment produced a non-finite transform")
        aligned = cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)
        alignment_ms = (time.perf_counter() - started) * 1000
        embedding_started = time.perf_counter()
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False
        )
        vector = self.session.run([self.output.name], {self.input.name: blob})[0].reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("AdaFace returned an invalid embedding")
        self.last_timings_ms = {
            "alignment": alignment_ms,
            "embedding": (time.perf_counter() - embedding_started) * 1000,
        }
        return vector / norm


class AdaFaceGalleryMatcher:
    """Open-set matcher using the independently calibrated 50-way AdaFace rule."""
    cache_key = "scrfd_adaface_v1"

    def __init__(self):
        self.recognition_threshold = ADAFACE_SIMILARITY_THRESHOLD
        self.margin_threshold = ADAFACE_MARGIN_THRESHOLD

    def match(self, query_embedding, gallery):
        names = gallery["names"]
        matrix = np.asarray(gallery["embeddings"], dtype=np.float32)
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if not names or matrix.shape != (len(names), query.size):
            raise ValueError("Gallery is empty or AdaFace embedding dimensions do not match")
        qnorm = float(np.linalg.norm(query))
        norms = np.linalg.norm(matrix, axis=1)
        if qnorm <= 1e-12 or np.any(norms <= 1e-12):
            raise ValueError("Zero-length AdaFace embedding is invalid")
        scores = np.clip((matrix @ query) / (norms * qnorm), -1.0, 1.0)
        order = np.argsort(scores)[::-1]
        first = int(order[0]); best = float(scores[first])
        second = float(scores[int(order[1])]) if len(order) > 1 else -1.0
        margin = best - second
        accepted = best >= self.recognition_threshold and margin >= self.margin_threshold
        return {
            "identity": names[first] if accepted else None,
            "status": "recognised" if accepted else "unknown",
            "best_score": best, "second_score": second, "margin": float(margin),
            "candidates": [{"name": names[int(i)], "score": float(scores[i])} for i in order[:3]],
        }

    @staticmethod
    def duplicate_status(result):
        return "unknown"
