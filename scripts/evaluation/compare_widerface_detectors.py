"""Compare YuNet and SCRFD-10G-KPS on a fixed WIDER FACE manifest.

This is a custom IoU evaluation, not the official WIDER FACE AP benchmark.
The default pilot uses the first 25 images in the existing 500-image manifest;
it never resamples images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for _path in (str(SCRIPTS_ROOT), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation.common import calculate_detection_metrics, validate_output_path, write_manifest_exclusive
from evaluation.evaluate_widerface import calculate_iou
from evaluation.widerface_common import (
    ANNOTATION_FILE,
    DATASET_ROOT,
    EVALUATION_OUTPUT_DIR,
    IMAGE_ROOT,
    load_annotations,
    validate_image_path,
)
from face_engine.detector import FaceDetector, MODEL_PATH as YUNET_MODEL_PATH


DEFAULT_MANIFEST = EVALUATION_OUTPUT_DIR / "widerface_mixed_500_seed42.json"
DEFAULT_SCRFD_CHECKPOINT = PROJECT_ROOT / "data/private/models/scrfd_10g_bnkps.onnx"
DEFAULT_RETINAFACE_CHECKPOINT = PROJECT_ROOT / "data/private/models/retinaface_int.onnx"
IOU_THRESHOLD = 0.5
MAX_DIMENSION = 1280
RETINAFACE_INPUT_HEIGHT = 608
RETINAFACE_INPUT_WIDTH = 640
RETINAFACE_ANCHOR_COUNT = 15960
RETINAFACE_CONFIG = {
    "min_sizes": ((16, 32), (64, 128), (256, 512)),
    "steps": (8, 16, 32),
    "variance": (0.1, 0.2),
    "clip": False,
}
FACE_HEIGHT_GROUPS = (
    ("le_20_px", 0.0, 20.0),
    ("gt_20_to_40_px", 20.0, 40.0),
    ("gt_40_px", 40.0, math.inf),
)


def _resolved_inside(path, root, label):
    path = Path(path)
    resolved = path.resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside {resolved_root}: {resolved}") from error
    return resolved


def load_fixed_manifest(manifest_path, limit=25):
    """Read and validate the existing 500-image manifest without resampling."""
    if limit < 1 or limit > 500:
        raise ValueError("--limit must be between 1 and 500; use 500 for the complete manifest.")
    manifest_path = _resolved_inside(
        manifest_path, EVALUATION_OUTPUT_DIR, "Input manifest"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"WIDER FACE input manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest.get("image_root", "")).resolve() != IMAGE_ROOT.resolve():
        raise ValueError("Input manifest image_root does not match the configured WIDER FACE image root.")
    if Path(manifest.get("annotation_file", "")).resolve() != ANNOTATION_FILE.resolve():
        raise ValueError("Input manifest annotation_file does not match the configured WIDER FACE annotations.")

    records = manifest.get("sampled_images")
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError("Expected the saved mixed-density manifest to contain exactly 500 sampled image records.")
    if manifest.get("sampled_image_count") != 500:
        raise ValueError("Input manifest sampled_image_count must be 500.")

    paths = []
    normalized = set()
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "evaluated":
            raise ValueError("Every manifest image must be an evaluated record.")
        raw_path = record.get("image")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Manifest image records must contain a relative image path.")
        relative_path = PurePosixPath(raw_path.replace("\\", "/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe image path in manifest: {raw_path}")
        key = relative_path.as_posix()
        if key in normalized:
            raise ValueError(f"Duplicate image path in manifest: {raw_path}")
        normalized.add(key)
        validate_image_path(key)
        paths.append((key, record))

    return manifest_path, manifest, paths[:limit]


def resize_preserving_aspect(image, max_dimension=MAX_DIMENSION):
    """Resize to the shared maximum and pad to stride 32 without distortion."""
    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")
    height, width = image.shape[:2]
    scale = min(max_dimension / max(width, height), 1.0)
    if scale == 1.0:
        resized = image
    else:
        size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    resized_height, resized_width = resized.shape[:2]
    padded_width = int(math.ceil(resized_width / 32.0) * 32)
    padded_height = int(math.ceil(resized_height / 32.0) * 32)
    if (padded_width, padded_height) == (resized_width, resized_height):
        return resized, scale
    padded = np.zeros((padded_height, padded_width, image.shape[2]), dtype=image.dtype)
    padded[:resized_height, :resized_width] = resized
    return padded, scale


def _box_height_group(box):
    height = float(box[3]) - float(box[1])
    if height <= 20:
        return "le_20_px"
    if height <= 40:
        return "gt_20_to_40_px"
    return "gt_40_px"


def validate_annotation_matches(selected, annotation_map):
    """Require every selected manifest path and its recorded face count to match annotations."""
    for image_index, (image_path, manifest_record) in enumerate(selected, start=1):
        if image_path not in annotation_map:
            raise ValueError(f"Manifest image has no annotation entry: {image_path}")
        if len(annotation_map[image_path]) != manifest_record.get("ground_truth_faces"):
            raise ValueError(f"Manifest/annotation face count mismatch for {image_path}.")


def match_image_predictions(gt_boxes, predictions, iou_threshold=IOU_THRESHOLD):
    """Apply the existing evaluator's confidence-ordered greedy IoU matching."""
    ordered = sorted(predictions, key=lambda row: float(row["confidence"]), reverse=True)
    matched_gt = set()
    matched_groups = {name: 0 for name, _, _ in FACE_HEIGHT_GROUPS}
    false_positives = 0
    for prediction in ordered:
        best_iou = 0.0
        best_gt_index = None
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            overlap = calculate_iou(prediction["bbox"], gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_index)
            matched_groups[_box_height_group(gt_boxes[best_gt_index])] += 1
        else:
            false_positives += 1

    fn_groups = {name: 0 for name, _, _ in FACE_HEIGHT_GROUPS}
    gt_groups = {name: 0 for name, _, _ in FACE_HEIGHT_GROUPS}
    for gt_index, box in enumerate(gt_boxes):
        group = _box_height_group(box)
        gt_groups[group] += 1
        if gt_index not in matched_gt:
            fn_groups[group] += 1

    return {
        "true_positives": len(matched_gt),
        "false_positives": false_positives,
        "false_negatives": len(gt_boxes) - len(matched_gt),
        "matched_ground_truth_by_height": matched_groups,
        "false_negatives_by_height": fn_groups,
        "ground_truth_by_height": gt_groups,
    }


class YuNetAdapter:
    def __init__(self, score_threshold=0.5, nms_threshold=0.3, max_dimension=MAX_DIMENSION):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_dimension = max_dimension
        self.detector = FaceDetector(
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_dimension=max_dimension,
        )

    def detect(self, image):
        rows = self.detector.detect(image)
        return [{
            "bbox": [x, y, x + width, y + height],
            "confidence": float(row["confidence"]),
            "landmarks": row["landmarks"],
        } for row in rows for x, y, width, height in [row["bbox"]]]


class SCRFDKPSAdapter:
    """Normalize InsightFace SCRFD KPS outputs; fail if five points are absent."""

    def __init__(self, model, score_threshold=0.5, nms_threshold=0.4):
        if not bool(getattr(model, "use_kps", False)):
            raise ValueError("SCRFD checkpoint does not expose five-point KPS outputs.")
        self.model = model
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image):
        height, width = image.shape[:2]
        boxes, keypoints = self.model.detect(
            image, input_size=(width, height), max_num=0, metric="default"
        )
        if boxes is None or len(boxes) == 0:
            return []
        if keypoints is None:
            raise RuntimeError("SCRFD checkpoint did not return KPS landmarks; use the five-landmark KPS variant.")

        boxes = np.asarray(boxes)
        keypoints = np.asarray(keypoints)
        if boxes.ndim != 2 or boxes.shape[1] < 5:
            raise RuntimeError("SCRFD returned malformed boxes; expected x1,y1,x2,y2,confidence.")
        if keypoints.shape == (len(boxes), 10):
            keypoints = keypoints.reshape((-1, 5, 2))
        if keypoints.shape != (len(boxes), 5, 2):
            raise RuntimeError("SCRFD KPS output must contain five 2D landmarks per detection.")

        normalized = []
        for box, points in zip(boxes, keypoints):
            values = np.concatenate((box[:5], points.reshape(-1)))
            if not np.isfinite(values).all():
                raise RuntimeError("SCRFD returned non-finite detection values.")
            if float(box[4]) < self.score_threshold:
                continue
            x1, y1, x2, y2, confidence = map(float, box[:5])
            normalized.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "landmarks": points.reshape(-1).astype(float).tolist(),
            })
        return normalized


def retinaface_preprocess(image):
    """Apply AMD RetinaFace's aspect-preserving BGR preprocessing and NHWC layout."""
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RetinaFace expects a non-empty three-channel BGR image.")
    height, width = image.shape[:2]
    target_ratio = RETINAFACE_INPUT_HEIGHT / RETINAFACE_INPUT_WIDTH
    if height / width <= target_ratio:
        resize_ratio = RETINAFACE_INPUT_WIDTH / float(width)
        resized_height = int(height * resize_ratio)
        resized_width = RETINAFACE_INPUT_WIDTH
    else:
        resize_ratio = RETINAFACE_INPUT_HEIGHT / float(height)
        resized_height = RETINAFACE_INPUT_HEIGHT
        resized_width = int(width * resize_ratio)
    if resized_height < 1 or resized_width < 1:
        raise ValueError("RetinaFace resize produced an empty image.")

    resized = cv2.resize(image, (resized_width, resized_height)).astype(np.float32)
    padded = np.zeros(
        (RETINAFACE_INPUT_HEIGHT, RETINAFACE_INPUT_WIDTH, 3), dtype=np.float32
    )
    padded[:resized_height, :resized_width] = resized
    padded -= np.asarray((104, 117, 123), dtype=np.float32)
    nchw = np.transpose(padded, (2, 0, 1))[None, ...]
    nhwc = np.transpose(nchw, (0, 2, 3, 1)).astype(np.float32, copy=False)
    return nhwc, float(resize_ratio)


def retinaface_prior_boxes(height=RETINAFACE_INPUT_HEIGHT,
                           width=RETINAFACE_INPUT_WIDTH):
    """Generate AMD RetinaFace MobileNet0.25 priors in the reference anchor order."""
    anchors = []
    for step, sizes in zip(RETINAFACE_CONFIG["steps"], RETINAFACE_CONFIG["min_sizes"]):
        feature_height = int(math.ceil(height / step))
        feature_width = int(math.ceil(width / step))
        for row in range(feature_height):
            for column in range(feature_width):
                for size in sizes:
                    anchors.append((
                        (column + 0.5) * step / width,
                        (row + 0.5) * step / height,
                        size / width,
                        size / height,
                    ))
    priors = np.asarray(anchors, dtype=np.float32)
    if priors.shape != (RETINAFACE_ANCHOR_COUNT, 4):
        raise RuntimeError(f"Unexpected RetinaFace prior shape: {priors.shape}")
    return priors


def _retinaface_nms(detections, threshold):
    """AMD utils.py py_cpu_nms, including its +1 area convention."""
    if not len(detections):
        return []
    x1, y1, x2, y2, scores = (detections[:, index] for index in range(5))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        overlap = intersection / (areas[index] + areas[order[1:]] - intersection)
        remaining = np.where(overlap <= threshold)[0]
        order = order[remaining + 1]
    return keep


def decode_retinaface_outputs(outputs, resize_ratio, original_width, original_height,
                              confidence_threshold=0.4, nms_threshold=0.4):
    """Decode AMD's loc/conf/landmark outputs into the shared adapter contract."""
    if not np.isfinite(resize_ratio) or resize_ratio <= 0:
        raise ValueError("RetinaFace resize ratio must be finite and positive.")
    if len(outputs) != 3:
        raise RuntimeError("RetinaFace must return loc, confidence, and landmark tensors.")
    loc, confidence, landmark_offsets = (np.asarray(output) for output in outputs)
    expected = (
        (1, RETINAFACE_ANCHOR_COUNT, 4),
        (1, RETINAFACE_ANCHOR_COUNT, 2),
        (1, RETINAFACE_ANCHOR_COUNT, 10),
    )
    actual = (loc.shape, confidence.shape, landmark_offsets.shape)
    if actual != expected:
        raise RuntimeError(
            f"Malformed RetinaFace outputs; expected {expected}, got {actual}."
        )
    if any(output.dtype != np.float32 for output in (loc, confidence, landmark_offsets)):
        raise RuntimeError("RetinaFace outputs must be float32 tensors.")
    if not all(np.isfinite(output).all() for output in (loc, confidence, landmark_offsets)):
        raise RuntimeError("RetinaFace returned non-finite output values.")

    loc = loc[0]
    confidence = confidence[0]
    landmark_offsets = landmark_offsets[0]
    priors = retinaface_prior_boxes()

    # AMD utils.py: softmax over class logits, class index 1 is face.
    shifted = confidence - confidence.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    scores = probabilities[:, 1]

    # AMD utils.py decode(): center/size offsets -> normalized xyxy boxes.
    variances = RETINAFACE_CONFIG["variance"]
    centers = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    box_scale = np.asarray(
        (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT,
         RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT), dtype=np.float32
    )
    boxes = boxes * box_scale / np.float32(resize_ratio)

    # AMD utils.py decode_landm(): five ordered (x, y) pairs.
    points = landmark_offsets.reshape((-1, 5, 2)).copy()
    for point_index in range(5):
        points[:, point_index, :] = (
            priors[:, :2]
            + points[:, point_index, :] * variances[0] * priors[:, 2:]
        )
    point_scale = np.asarray(
        (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT) * 5, dtype=np.float32
    )
    points = points.reshape((-1, 10)) * point_scale / np.float32(resize_ratio)

    selected = np.where(scores > confidence_threshold)[0]
    boxes, scores, points = boxes[selected], scores[selected], points[selected]
    order = scores.argsort()[::-1]
    boxes, scores, points = boxes[order], scores[order], points[order]
    detections = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
    keep = _retinaface_nms(detections, nms_threshold)
    detections, points = detections[keep], points[keep]

    predictions = []
    for detection, landmarks in zip(detections, points):
        values = np.concatenate((detection, landmarks))
        if not np.isfinite(values).all():
            raise RuntimeError("RetinaFace decoding produced non-finite detections.")
        x1, y1, x2, y2, score = map(float, detection)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("RetinaFace decoding produced a non-positive box.")
        predictions.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": score,
            "landmarks": landmarks.astype(float).tolist(),
        })
    return predictions


class RetinaFaceAdapter:
    """AMD RetinaFace MobileNet0.25 ONNX adapter using CPUExecutionProvider."""

    def __init__(self, session, confidence_threshold=0.4, nms_threshold=0.4):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("RetinaFace confidence threshold must be between 0 and 1.")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("RetinaFace NMS threshold must be between 0 and 1.")
        providers = session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise ValueError(f"RetinaFace session must use CPUExecutionProvider only; got {providers}.")
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise RuntimeError("RetinaFace expects one float32 ONNX input.")
        if len(inputs[0].shape) != 4 or inputs[0].shape[1:] != [608, 640, 3]:
            raise RuntimeError(
                f"RetinaFace ONNX input must be NHWC [N,608,640,3], got {inputs[0].shape}."
            )
        expected_outputs = ((15960, 4), (15960, 2), (15960, 10))
        if len(outputs) != 3:
            raise RuntimeError("RetinaFace ONNX model must expose loc, confidence, and landmark outputs.")
        for output, shape in zip(outputs, expected_outputs):
            if output.type != "tensor(float)" or len(output.shape) != 3:
                raise RuntimeError(f"Malformed RetinaFace output metadata: {output.name} {output.shape}.")
            if output.shape[1:] != list(shape):
                raise RuntimeError(
                    f"RetinaFace output {output.name} expected [N,{shape[0]},{shape[1]}], "
                    f"got {output.shape}."
                )
        self.session = session
        self.input_name = inputs[0].name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image):
        tensor, resize_ratio = retinaface_preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        return decode_retinaface_outputs(
            outputs, resize_ratio, image.shape[1], image.shape[0],
            self.confidence_threshold, self.nms_threshold,
        )


def create_retinaface_adapter(checkpoint_path, confidence_threshold=0.4,
                              nms_threshold=0.4):
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RetinaFace checkpoint not found: {checkpoint_path}")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("RetinaFace evaluation requires ONNX Runtime.") from error
    session = ort.InferenceSession(
        str(checkpoint_path), providers=["CPUExecutionProvider"]
    )
    return RetinaFaceAdapter(session, confidence_threshold, nms_threshold)


def create_scrfd_adapter(checkpoint_path, score_threshold=0.5,
                         nms_threshold=0.4, max_dimension=MAX_DIMENSION):
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SCRFD KPS checkpoint not found: {checkpoint_path}")
    try:
        from insightface.model_zoo.scrfd import SCRFD
    except ImportError as error:
        raise RuntimeError("SCRFD evaluation requires the InsightFace package.") from error

    model = SCRFD(str(checkpoint_path))
    model.prepare(
        ctx_id=-1,
        det_thresh=score_threshold,
        nms_thresh=nms_threshold,
        input_size=(max_dimension, max_dimension),
    )
    static_input_size = getattr(model, "static_input_size", None)
    if static_input_size is not None:
        raise RuntimeError(
            "SCRFD checkpoint must support dynamic input sizes for the common aspect-preserving image tensor."
        )
    return SCRFDKPSAdapter(model, score_threshold, nms_threshold)


def _latency_summary(samples_ms):
    if not samples_ms:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _empty_counts():
    return {"images_attempted": 0, "images_evaluated": 0, "unreadable_images": 0,
            "unreadable_ground_truth_faces_excluded": 0,
            "detector_exceptions": 0, "predictions": 0, "true_positives": 0,
            "false_positives": 0, "false_negatives": 0,
            "failure_categories": {name: 0 for name in (
                "image_read_failure", "preprocessing_exception", "detector_exception",
                "prediction_validation_exception", "matching_exception",
            )},
            "matched_ground_truth_by_height": {name: 0 for name, _, _ in FACE_HEIGHT_GROUPS},
            "false_negatives_by_height": {name: 0 for name, _, _ in FACE_HEIGHT_GROUPS},
            "ground_truth_by_height": {name: 0 for name, _, _ in FACE_HEIGHT_GROUPS}}


def _add_match_counts(counts, matches):
    for key in ("true_positives", "false_positives", "false_negatives"):
        counts[key] += matches[key]
    for key in ("matched_ground_truth_by_height", "false_negatives_by_height", "ground_truth_by_height"):
        for group, value in matches[key].items():
            counts[key][group] += value


def _count_exception_as_misses(counts, gt_boxes):
    """Count readable-image detector exceptions as FNs rather than hiding GT."""
    counts["false_negatives"] += len(gt_boxes)
    for box in gt_boxes:
        group = _box_height_group(box)
        counts["ground_truth_by_height"][group] += 1
        counts["false_negatives_by_height"][group] += 1


def _read_evaluation_image(image_file):
    """Return the image and a structured failure if OpenCV cannot read it."""
    try:
        image = cv2.imread(str(image_file))
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    if image is None:
        return None, "cv2.imread returned None"
    return image, None


def _record_image_read_failure(result, row, error, gt_boxes):
    counts = result["counts"]
    counts["unreadable_images"] += 1
    counts["unreadable_ground_truth_faces_excluded"] += len(gt_boxes)
    counts["failure_categories"]["image_read_failure"] += 1
    row.update({"status": "excluded", "reason": "image_read_failure", "error": str(error)})


def _record_processing_failure(result, row, category, error, gt_boxes, started):
    counts = result["counts"]
    counts["failure_categories"][category] += 1
    if category == "detector_exception":
        counts["detector_exceptions"] += 1
    _count_exception_as_misses(counts, gt_boxes)
    latency_ms = (time.perf_counter() - started) * 1000.0
    result["latency_samples_ms"].append(latency_ms)
    row.update({"status": "failed", "reason": category, "error": str(error),
                "latency_ms": latency_ms, "false_negatives": len(gt_boxes)})


def _evaluate_detector_image(result, image, image_path, gt_boxes, adapter, max_dimension):
    """Run one detector and attribute failures to the stage that raised them."""
    counts = result["counts"]
    row = {"image": image_path, "ground_truth_faces": len(gt_boxes)}
    started = time.perf_counter()
    try:
        prepared, scale = resize_preserving_aspect(image, max_dimension)
    except Exception as error:
        _record_processing_failure(
            result, row, "preprocessing_exception", error, gt_boxes, started
        )
        return row

    try:
        predictions = adapter.detect(prepared)
    except Exception as error:
        _record_processing_failure(
            result, row, "detector_exception", error, gt_boxes, started
        )
        return row

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not math.isfinite(latency_ms):
        _record_processing_failure(
            result, row, "detector_exception", "Non-finite inference latency.",
            gt_boxes, started,
        )
        return row

    try:
        normalized_predictions = []
        for prediction in predictions:
            box = [float(value) / scale for value in prediction["bbox"]]
            confidence = float(prediction["confidence"])
            if len(box) != 4 or not np.isfinite(box).all() or not math.isfinite(confidence):
                raise RuntimeError("Detector returned invalid box or confidence values.")
            normalized_predictions.append({"bbox": box, "confidence": confidence})
    except Exception as error:
        _record_processing_failure(
            result, row, "prediction_validation_exception", error, gt_boxes, started
        )
        return row

    try:
        matches = match_image_predictions(gt_boxes, normalized_predictions)
    except Exception as error:
        _record_processing_failure(
            result, row, "matching_exception", error, gt_boxes, started
        )
        return row

    counts["images_evaluated"] += 1
    counts["predictions"] += len(normalized_predictions)
    _add_match_counts(counts, matches)
    result["latency_samples_ms"].append(latency_ms)
    row.update({"status": "evaluated", "predictions": len(normalized_predictions),
                "latency_ms": latency_ms, **matches})
    return row


def _finalize_detector_result(result):
    counts = result["counts"]
    result["custom_metrics"] = calculate_detection_metrics(
        counts["true_positives"], counts["false_positives"], counts["false_negatives"]
    )
    result["recall_by_face_height"] = {
        group: {
            "ground_truth_faces": counts["ground_truth_by_height"][group],
            "true_positives": counts["matched_ground_truth_by_height"][group],
            "false_negatives": counts["false_negatives_by_height"][group],
            "recall": (
                counts["matched_ground_truth_by_height"][group]
                / counts["ground_truth_by_height"][group]
                if counts["ground_truth_by_height"][group] else 0.0
            ),
        }
        for group, _, _ in FACE_HEIGHT_GROUPS
    }
    result["latency"] = _latency_summary(result.pop("latency_samples_ms"))
    return result


def run_comparison(manifest_path, output_dir, scrfd_checkpoint,
                   limit=25, yunet_score_threshold=0.5, yunet_nms_threshold=0.3,
                   scrfd_score_threshold=0.5, scrfd_nms_threshold=0.4,
                   max_dimension=MAX_DIMENSION, retinaface_checkpoint=None,
                   retinaface_score_threshold=0.4, retinaface_nms_threshold=0.4):
    output_dir = Path(output_dir)
    protected_roots = (DATASET_ROOT, ANNOTATION_FILE, manifest_path)
    if retinaface_checkpoint is not None:
        protected_roots += (retinaface_checkpoint,)
    safe_output = validate_output_path(
        output_dir / "detector_comparison.json",
        EVALUATION_OUTPUT_DIR,
        protected_roots=protected_roots,
    )
    output_dir = safe_output.parent
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing evaluation directory: {output_dir}")
    if max_dimension != MAX_DIMENSION:
        raise ValueError("This comparison fixes the common maximum image dimension at 1280.")
    for label, value in (
        ("YuNet confidence threshold", yunet_score_threshold),
        ("YuNet NMS threshold", yunet_nms_threshold),
        ("SCRFD confidence threshold", scrfd_score_threshold),
        ("SCRFD NMS threshold", scrfd_nms_threshold),
        ("RetinaFace confidence threshold", retinaface_score_threshold),
        ("RetinaFace NMS threshold", retinaface_nms_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be between 0 and 1.")

    manifest_path, source_manifest, selected = load_fixed_manifest(manifest_path, limit)
    annotations = load_annotations()
    annotation_map = {}
    for image_path, boxes in annotations:
        if image_path in annotation_map:
            raise ValueError(f"Duplicate image path in WIDER FACE annotations: {image_path}")
        annotation_map[image_path] = boxes
    validate_annotation_matches(selected, annotation_map)

    if not Path(YUNET_MODEL_PATH).is_file():
        raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL_PATH}")
    detectors = {
        "yunet": {
            "adapter": YuNetAdapter(yunet_score_threshold, yunet_nms_threshold, max_dimension),
            "model": Path(YUNET_MODEL_PATH).name,
            "confidence_threshold": yunet_score_threshold,
            "nms_threshold": yunet_nms_threshold,
        },
        "scrfd_10g_kps": {
            "adapter": create_scrfd_adapter(
                scrfd_checkpoint, scrfd_score_threshold, scrfd_nms_threshold, max_dimension
            ),
            "model": Path(scrfd_checkpoint).name,
            "confidence_threshold": scrfd_score_threshold,
            "nms_threshold": scrfd_nms_threshold,
        },
    }
    if retinaface_checkpoint is not None:
        retinaface_checkpoint = Path(retinaface_checkpoint).resolve()
        detectors["retinaface_mobilenet025"] = {
            "adapter": create_retinaface_adapter(
                retinaface_checkpoint,
                retinaface_score_threshold,
                retinaface_nms_threshold,
            ),
            "model": retinaface_checkpoint.name,
            "confidence_threshold": retinaface_score_threshold,
            "nms_threshold": retinaface_nms_threshold,
        }
    detectors["yunet"]["comparison_label"] = "YuNet (common comparison preprocessing)"
    detectors["scrfd_10g_kps"]["comparison_label"] = "SCRFD-10G-KPS (common comparison preprocessing)"
    if "retinaface_mobilenet025" in detectors:
        detectors["retinaface_mobilenet025"]["comparison_label"] = (
            "AMD RetinaFace MobileNet 0.25 (common comparison input; AMD model preprocessing)"
        )
    results = {
        name: {"model": config["model"], "comparison_label": config["comparison_label"],
               "settings": {"confidence_threshold": config["confidence_threshold"],
                            "nms_threshold": config["nms_threshold"],
                            "max_dimension": max_dimension,
                            "resize_policy": "aspect-preserving; OpenCV INTER_AREA when downsampling",
                            "comparison_preprocessing": (
                                 "Shared aspect-preserving resize and bottom/right padding to stride 32; "
                                 "YuNet result uses this common preprocessing and differs from the "
                                 "historical unpadded evaluator."
                                 if name != "retinaface_mobilenet025" else
                                 "Shared aspect-preserving resize and bottom/right padding to stride 32, "
                                 "then AMD RetinaFace preprocessing: aspect-preserving resize and "
                                 "bottom/right zero padding to 608x640, BGR mean subtraction, NHWC."
                            )},
               "counts": _empty_counts(), "latency_samples_ms": [], "per_image": []}
        for name, config in detectors.items()
    }

    image_read_failures = []
    run_started = time.perf_counter()

    for image_index, (image_path, manifest_record) in enumerate(selected, start=1):
        image_file = validate_image_path(image_path)
        image, read_error = _read_evaluation_image(image_file)
        if read_error is not None:
            image_read_failures.append({"image": image_path, "error": read_error})
        for name, config in detectors.items():
            result = results[name]
            counts = result["counts"]
            counts["images_attempted"] += 1
            row = {"image": image_path, "ground_truth_faces": len(annotation_map[image_path])}
            if read_error is not None:
                _record_image_read_failure(result, row, read_error, annotation_map[image_path])
                result["per_image"].append(row)
                continue
            result["per_image"].append(_evaluate_detector_image(
                result, image, image_path, annotation_map[image_path],
                config["adapter"], max_dimension,
            ))
        elapsed = time.perf_counter() - run_started
        print(
            f"Progress: {image_index}/{len(selected)} images "
            f"({image_path}); {elapsed:.1f}s elapsed",
            file=sys.stderr,
            flush=True,
        )

    finalized = {name: _finalize_detector_result(value) for name, value in results.items()}
    report = {
        "evaluation_type": "custom WIDER FACE fixed-manifest detector comparison",
        "official_benchmark_ap": False,
        "method_note": (
            "Custom confidence-ordered greedy one-to-one matching at IoU >= 0.5. "
            "These precision/recall/F1 results are not official WIDER FACE AP."
        ),
        "denominator_policy": (
            "Images whose read returns None or raises are image_read_failure cases; their ground-truth "
            "boxes are excluded for both detectors and reported separately. Failures after a successful "
            "read are categorized by stage and contribute all ground-truth faces as false negatives."
        ),
        "image_read_failures": image_read_failures,
        "input_manifest": str(manifest_path),
        "source_manifest_seed": source_manifest.get("seed"),
        "annotation_file": str(ANNOTATION_FILE),
        "image_root": str(IMAGE_ROOT),
        "requested_pilot_count": limit,
        "selected_images": [path for path, _ in selected],
        "iou_threshold": IOU_THRESHOLD,
        "latency_policy": {
            "label": "cold-inclusive per-image latency",
            "includes": [
                "aspect-preserving resize and stride padding",
                "detector preprocessing, inference, and detector postprocessing",
                "lazy runtime/session initialization incurred during the measured call, if any",
            ],
            "excludes": [
                "disk image loading",
                "prediction normalization and validation",
                "IoU matching",
                "report writing",
            ],
            "interpretation": (
                "The first inference is included. This is cold-inclusive per-image latency and does "
                "not claim steady-state inference performance. Successful calls and processing failures "
                "after image loading contribute latency samples; image-read failures do not."
            ),
        },
        "face_height_groups": {"le_20_px": "height <= 20 px",
                               "gt_20_to_40_px": "20 px < height <= 40 px",
                               "gt_40_px": "height > 40 px"},
        "detectors": finalized,
    }

    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-",
                                             dir=EVALUATION_OUTPUT_DIR))
    published = False
    try:
        write_manifest_exclusive(temporary_output / "detector_comparison.json", report)
        safe_output = validate_output_path(
            output_dir / "detector_comparison.json",
            EVALUATION_OUTPUT_DIR,
            protected_roots=protected_roots,
        )
        output_dir = safe_output.parent
        if output_dir.exists():
            raise FileExistsError(f"Refusing to reuse existing evaluation directory: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temporary_output, output_dir)
        published = True
    finally:
        if not published and temporary_output.exists():
            shutil.rmtree(temporary_output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scrfd-checkpoint", type=Path, default=DEFAULT_SCRFD_CHECKPOINT,
                        help="SCRFD-10G KPS ONNX checkpoint; ordinary non-KPS checkpoints are rejected.")
    parser.add_argument("--limit", type=int, default=25,
                        help="Number of leading images from the saved manifest; never resamples.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--yunet-score-threshold", type=float, default=0.5)
    parser.add_argument("--yunet-nms-threshold", type=float, default=0.3)
    parser.add_argument("--scrfd-score-threshold", type=float, default=0.5)
    parser.add_argument("--scrfd-nms-threshold", type=float, default=0.4)
    parser.add_argument(
        "--retinaface-checkpoint", type=Path, default=None,
        help="Optional AMD RetinaFace MobileNet 0.25 ONNX checkpoint; when supplied, adds this detector.",
    )
    parser.add_argument("--retinaface-score-threshold", type=float, default=0.4)
    parser.add_argument("--retinaface-nms-threshold", type=float, default=0.4)
    args = parser.parse_args(argv)
    if args.iou != IOU_THRESHOLD:
        parser.error("This pilot fixes IoU at 0.5 to match the existing WIDER FACE evaluator.")
    report = run_comparison(
        args.input_manifest, args.output_dir, args.scrfd_checkpoint, args.limit,
        args.yunet_score_threshold, args.yunet_nms_threshold,
        args.scrfd_score_threshold, args.scrfd_nms_threshold, MAX_DIMENSION,
        args.retinaface_checkpoint, args.retinaface_score_threshold,
        args.retinaface_nms_threshold,
    )
    print(json.dumps({"output_dir": str(args.output_dir),
                      "detectors": {name: {"counts": value["counts"],
                                            "custom_metrics": value["custom_metrics"],
                                            "latency": value["latency"]}
                                    for name, value in report["detectors"].items()}}, indent=2))


if __name__ == "__main__":
    main()
