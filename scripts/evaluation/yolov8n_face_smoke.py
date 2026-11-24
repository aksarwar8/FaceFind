"""Isolated YOLOv8n-Face ONNX WIDER FACE smoke test (CPU only)."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data/private/models/yolov8n-face.onnx"
MANIFEST_PATH = PROJECT_ROOT / "data/private/evaluation/widerface_mixed_500_seed42.json"
OUTPUT_DIR = PROJECT_ROOT / "data/private/evaluation/yolov8n_face_smoke_20261123"
MODEL_SOURCE = (
    "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/"
    "weights/yolov8n-face.onnx"
)
CODE_SOURCE = "https://github.com/yakhyo/yolov8-face-onnx-inference"
INPUT_SIZE = 640
STRIDES = (8, 16, 32)


def letterbox(image: np.ndarray, size: int = INPUT_SIZE):
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width, resized_height = int(width * scale), int(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - resized_width) / 2.0, (size - resized_height) / 2.0
    left, top = int(pad_x), int(pad_y)
    right, bottom = size - resized_width - left, size - resized_height - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, pad_x, pad_y


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float):
    """Score ordered NMS matching the inspected upstream NumPy implementation."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = (boxes[:, i] for i in range(4))
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        if not rest.size:
            break
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        denominator = areas[current] + areas[rest] - intersection
        overlap = np.divide(
            intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0
        )
        order = rest[np.where(overlap <= threshold)[0]]
    return np.asarray(keep, dtype=np.int64)


class YOLOv8FaceONNXAdapter:
    """Decode the upstream model's three stride heads to FaceFind detection rows."""

    def __init__(self, model_path: Path, confidence_threshold=0.5, nms_threshold=0.45,
                 max_detections=750):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise RuntimeError("Expected one float32 YOLO input.")
        if len(outputs) != 3 or any(output.type != "tensor(float)" for output in outputs):
            raise RuntimeError("Expected three float32 YOLO stride-head outputs.")
        input_shape = inputs[0].shape
        if len(input_shape) != 4 or input_shape[1] != 3:
            raise RuntimeError(f"Expected dynamic NCHW RGB input with 3 channels; got {input_shape}.")
        self.input_name = inputs[0].name
        self.output_metadata = outputs
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.last_candidate_count = 0
        self.last_output_shapes = []

    @staticmethod
    def _softmax(values, axis=-1):
        exp_values = np.exp(values - np.max(values, axis=axis, keepdims=True))
        return exp_values / np.sum(exp_values, axis=axis, keepdims=True)

    def detect(self, image: np.ndarray):
        padded, scale, pad_x, pad_y = letterbox(image)
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        outputs = self.session.run(None, {self.input_name: tensor})
        boxes_all, scores_all, points_all = [], [], []
        observed_shapes = []

        for output, stride in zip(outputs, STRIDES):
            if output.ndim != 4 or output.shape[:2] != (1, 80):
                raise RuntimeError(f"Unexpected stride output shape: {output.shape}.")
            _, _, height, width = output.shape
            if (height, width) != (INPUT_SIZE // stride, INPUT_SIZE // stride):
                raise RuntimeError(f"Unexpected {stride}-stride feature map: {output.shape}.")
            observed_shapes.append(list(output.shape))
            pred = output.reshape(1, 80, -1).transpose(0, 2, 1)[0]
            grid_y, grid_x = np.meshgrid(
                np.arange(height) + 0.5, np.arange(width) + 0.5, indexing="ij"
            )
            grid_x, grid_y = grid_x.ravel(), grid_y.ravel()

            distances = self._softmax(pred[:, :64].reshape(-1, 4, 16), axis=-1) @ np.arange(16)
            boxes = np.stack((
                (grid_x - distances[:, 0]) * stride,
                (grid_y - distances[:, 1]) * stride,
                (grid_x + distances[:, 2]) * stride,
                (grid_y + distances[:, 3]) * stride,
            ), axis=-1)

            point_y, point_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            points = pred[:, 65:].reshape(-1, 5, 3)
            points = np.stack((
                (points[:, :, 0] * 2.0 + point_x.ravel()[:, None]) * stride,
                (points[:, :, 1] * 2.0 + point_y.ravel()[:, None]) * stride,
            ), axis=-1).reshape(-1, 10)
            logits = pred[:, 64]
            scores = np.empty_like(logits)
            positive = logits >= 0
            scores[positive] = 1 / (1 + np.exp(-logits[positive]))
            exp_logits = np.exp(logits[~positive])
            scores[~positive] = exp_logits / (1 + exp_logits)
            boxes_all.append(boxes)
            scores_all.append(scores)
            points_all.append(points)

        boxes = np.concatenate(boxes_all)
        scores = np.concatenate(scores_all)
        points = np.concatenate(points_all)
        mask = scores >= self.confidence_threshold
        boxes, scores, points = boxes[mask], scores[mask], points[mask]
        candidate_count = len(scores)
        if candidate_count:
            keep = numpy_nms(boxes, scores, self.nms_threshold)[: self.max_detections]
            boxes, scores, points = boxes[keep], scores[keep], points[keep]
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            points = np.empty((0, 10), dtype=np.float32)

        # Undo letterboxing and clip only boxes, as in the upstream utility.
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad_x) / scale
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad_y) / scale
        points[:, 0::2] = (points[:, 0::2] - pad_x) / scale
        points[:, 1::2] = (points[:, 1::2] - pad_y) / scale
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, image.shape[1])
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, image.shape[0])

        detections = []
        for box, score, landmarks in zip(boxes, scores, points):
            if not np.isfinite(box).all() or not math.isfinite(float(score)):
                raise RuntimeError("Non-finite box or confidence returned by YOLOv8-Face.")
            if landmarks.shape != (10,) or not np.isfinite(landmarks).all():
                raise RuntimeError("Expected five finite decoded (x,y) landmarks per detection.")
            detections.append({
                "bbox": box.astype(float).tolist(),
                "confidence": float(score),
                "landmarks": landmarks.astype(float).tolist(),
            })
        self.last_candidate_count = candidate_count
        self.last_output_shapes = observed_shapes
        return detections


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to reuse existing smoke-test directory: {OUTPUT_DIR}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLOv8n-Face model not found: {MODEL_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    image_rel = manifest["sampled_images"][0]["image"]
    image_path = PROJECT_ROOT / "data/benchmarks/widerface/WIDER_val/WIDER_val/images" / image_rel
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not load WIDER FACE validation image: {image_path}")

    adapter = YOLOv8FaceONNXAdapter(MODEL_PATH)
    started = time.perf_counter()
    detections = adapter.detect(image)
    candidate_count = adapter.last_candidate_count
    actual_output_shapes = adapter.last_output_shapes
    inference_latency_ms = (time.perf_counter() - started) * 1000.0
    if candidate_count and len(detections) > candidate_count:
        raise RuntimeError("NMS increased the number of candidates.")
    if not all(
        0 <= d["bbox"][0] <= d["bbox"][2] <= image.shape[1]
        and 0 <= d["bbox"][1] <= d["bbox"][3] <= image.shape[0]
        and len(d["landmarks"]) == 10
        for d in detections
    ):
        raise RuntimeError("Output clipping or landmark conversion validation failed.")

    duplicate_keep = numpy_nms(
        np.asarray([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float32),
        np.asarray([0.9, 0.8], dtype=np.float32), 0.45,
    )
    if len(duplicate_keep) != 1:
        raise RuntimeError("Synthetic duplicate-box NMS validation failed.")

    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{detection['confidence']:.2f}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        for x, y in np.asarray(detection["landmarks"]).reshape(5, 2):
            cv2.circle(annotated, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    image_output = OUTPUT_DIR / "widerface_yolov8n_face_smoke.jpg"
    if not cv2.imwrite(str(image_output), annotated):
        raise RuntimeError(f"Failed to save annotated smoke-test image: {image_output}")
    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    input_meta = adapter.session.get_inputs()[0]
    output_meta = adapter.output_metadata
    onnx_meta = adapter.session.get_modelmeta()
    report = {
        "experiment": "YOLOv8n-Face ONNX one-image CPU smoke test",
        "model": {
            "path": str(MODEL_PATH), "source": MODEL_SOURCE, "code_source": CODE_SOURCE,
            "size_bytes": MODEL_PATH.stat().st_size, "sha256": model_hash,
            "published_checksum": None,
            "published_checksum_note": "No authoritative checksum was published/found for this release asset.",
            "license_metadata": onnx_meta.custom_metadata_map.get("license"),
            "repository_license": "No LICENSE file present in the cited inference repository.",
            "model_metadata": onnx_meta.custom_metadata_map,
        },
        "runtime": {
            "python": sys.version.split()[0], "onnxruntime": ort.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__,
            "providers_requested": ["CPUExecutionProvider"],
            "providers_used": adapter.session.get_providers(),
        },
        "onnx": {
            "inputs": [{"name": input_meta.name, "shape": input_meta.shape,
                        "type": input_meta.type, "layout": "NCHW", "color": "RGB",
                        "normalization": "float32 / 255"}],
            "outputs": [{"name": x.name, "shape": x.shape, "type": x.type}
                        for x in output_meta],
            "actual_output_shapes_at_640": actual_output_shapes,
        },
        "image": {"manifest": str(MANIFEST_PATH), "relative_path": image_rel,
                  "absolute_path": str(image_path), "shape": list(image.shape)},
        "settings": {"input_size": INPUT_SIZE, "confidence_threshold": 0.5,
                     "nms_iou_threshold": 0.45, "max_detections": 750},
        "smoke_test": {
            "status": "passed", "detection_count": len(detections),
            "confidence_pass_candidates_before_nms": candidate_count,
            "detections_removed_by_nms": candidate_count - len(detections),
            "synthetic_duplicate_nms_kept": len(duplicate_keep),
            "all_boxes_finite_and_clipped": True,
            "all_confidences_finite": True,
            "five_landmarks_per_detection": True,
            "benchmark_detection_format": "list of {bbox: [x1,y1,x2,y2], confidence: float, landmarks: [x,y]*5}",
            "inference_latency_ms": inference_latency_ms,
            "latency_scope": "Preprocessing, ONNX CPU run, decoding, NMS, inverse letterbox, clipping; excludes session creation and disk image loading.",
            "detections": detections,
        },
        "artifacts": {"annotated_image": str(image_output),
                      "report": str(OUTPUT_DIR / "smoke_test_report.json")},
    }
    (OUTPUT_DIR / "smoke_test_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "output_dir": str(OUTPUT_DIR),
                      "detection_count": len(detections),
                      "latency_ms": inference_latency_ms, "sha256": model_hash,
                      "actual_output_shapes": actual_output_shapes}, indent=2))


if __name__ == "__main__":
    main()
