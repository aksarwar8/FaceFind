"""Rerun selected WIDER FACE error cases and save visual diagnostics.

These are targeted diagnostic reruns. They are not the predictions recorded by
the original evaluation, whose manifest did not preserve prediction boxes.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.widerface_common import (  # noqa: E402
    ANNOTATION_FILE,
    DATASET_ROOT,
    EVALUATION_OUTPUT_DIR,
    IMAGE_ROOT,
    load_annotations,
    validate_image_path,
)
from face_engine.detector import FaceDetector, MODEL_PATH as DETECTOR_MODEL_PATH  # noqa: E402

# These fallback values are the documented defaults in evaluate_widerface.py.
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_MAX_DIMENSION = 1280
DEFAULT_IOU_THRESHOLD = 0.5


def calculate_iou(box_a, box_b):
    """Calculate IoU for [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def validate_manifest_input_path(manifest_path):
    """Require the read-only input manifest to be a regular file under eval output."""
    output_root = Path(EVALUATION_OUTPUT_DIR)
    resolved_root = output_root.resolve()
    expected_root = Path(os.path.abspath(output_root))
    if resolved_root != expected_root:
        raise ValueError(
            f"Evaluation output directory must not resolve through a symlink: {output_root}"
        )

    path = Path(manifest_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved_path = path.resolve()
    if resolved_path != Path(os.path.abspath(path)):
        raise ValueError(f"Manifest path must not resolve through a symlink: {path}")
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Manifest must be inside the evaluation output directory: {resolved_path}"
        ) from error
    if resolved_path == resolved_root or not resolved_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {resolved_path}")
    if resolved_path.stat().st_nlink > 1:
        raise ValueError(f"Manifest path has multiple hard links and is unsafe: {resolved_path}")
    if resolved_path == Path(ANNOTATION_FILE).resolve() or Path(DATASET_ROOT).resolve() in resolved_path.parents:
        raise ValueError(f"Manifest path must be outside the WIDER FACE dataset: {resolved_path}")
    return resolved_path


def validate_diagnostic_output_dir(output_dir=None, manifest_path=None):
    """Validate a new, direct-child diagnostic directory without creating it."""
    output_root = Path(EVALUATION_OUTPUT_DIR)
    resolved_root = output_root.resolve()
    expected_root = Path(os.path.abspath(output_root))
    if resolved_root != expected_root:
        raise ValueError(
            f"Evaluation output directory must not resolve through a symlink: {output_root}"
        )
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Evaluation output directory not found: {resolved_root}")

    if output_dir is None:
        stem = Path(manifest_path).stem if manifest_path else "widerface"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = resolved_root / f"{stem}_diagnostics_{timestamp}_{uuid.uuid4().hex[:8]}"
    else:
        candidate = Path(output_dir)
        if not candidate.is_absolute():
            candidate = resolved_root / candidate

    absolute_candidate = Path(os.path.abspath(candidate))
    resolved_candidate = candidate.resolve()
    if resolved_candidate != absolute_candidate:
        raise ValueError(f"Diagnostic output path must not resolve through a symlink: {candidate}")
    if resolved_candidate.parent != resolved_root:
        raise ValueError(
            f"Diagnostic output directory must be a direct child of {resolved_root}: {candidate}"
        )
    if resolved_candidate == Path(DATASET_ROOT).resolve() or Path(DATASET_ROOT).resolve() in resolved_candidate.parents:
        raise ValueError(f"Diagnostic output must be outside the WIDER FACE dataset: {candidate}")
    if resolved_candidate.exists():
        raise FileExistsError(f"Refusing to reuse existing diagnostic output directory: {resolved_candidate}")
    return resolved_candidate


def detector_configuration(manifest):
    """Return recorded detector settings and explicit documented fallbacks."""
    recorded = manifest.get("detector_parameters") or {}
    if not isinstance(recorded, dict):
        raise ValueError("Manifest detector_parameters must be a JSON object.")
    defaults = {
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "nms_threshold": DEFAULT_NMS_THRESHOLD,
        "max_dimension": DEFAULT_MAX_DIMENSION,
    }
    configuration = {}
    sources = {}
    for name, default_value in defaults.items():
        if name in recorded:
            configuration[name] = recorded[name]
            sources[name] = "manifest"
        else:
            configuration[name] = default_value
            sources[name] = "evaluate_widerface.py documented default"
    iou_threshold = manifest.get("iou_threshold")
    if iou_threshold is None:
        iou_threshold = DEFAULT_IOU_THRESHOLD
        sources["iou_threshold"] = "evaluate_widerface.py documented default"
    else:
        sources["iou_threshold"] = "manifest"
    for name, value in configuration.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid detector parameter in manifest: {name}={value!r}")
        if name != "max_dimension" and not 0 <= value <= 1:
            raise ValueError(f"Detector parameter is out of range: {name}={value!r}")
        if name == "max_dimension" and value <= 0:
            raise ValueError(f"Detector parameter must be positive: {name}={value!r}")
        if name == "max_dimension" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("Manifest max_dimension must be an integer.")
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)) or not 0 < iou_threshold <= 1:
        raise ValueError(f"Invalid IoU threshold: {iou_threshold!r}")
    configuration["iou_threshold"] = float(iou_threshold)
    return configuration, sources


def select_error_cases(manifest, num_fn=10, num_fp=10, seed=None):
    """Select deterministic samples from image-level FN and FP case lists."""
    if num_fn < 0 or num_fp < 0:
        raise ValueError("Requested FN/FP case counts must be nonnegative.")
    rows = manifest.get("sampled_images")
    if not isinstance(rows, list):
        raise ValueError("Manifest is missing its sampled_images list.")
    if seed is None:
        seed = manifest.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Manifest/CLI seed must be an integer.")

    by_path = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "evaluated":
            continue
        image_path = row.get("image")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("Evaluated manifest row is missing its image path.")
        if image_path in by_path:
            raise ValueError(f"Duplicate image path in manifest: {image_path}")
        by_path[image_path] = row

    fn_candidates = sorted(
        (row for row in by_path.values() if _nonnegative_int(row, "false_negatives") > 0),
        key=lambda row: row["image"],
    )
    fp_candidates = sorted(
        (row for row in by_path.values() if _nonnegative_int(row, "false_positives") > 0),
        key=lambda row: row["image"],
    )
    rng = random.Random(seed)
    selected_fn = rng.sample(fn_candidates, min(num_fn, len(fn_candidates)))
    selected_fp = rng.sample(fp_candidates, min(num_fp, len(fp_candidates)))

    selected = {}
    for category, chosen in (("fn", selected_fn), ("fp", selected_fp)):
        for row in chosen:
            item = selected.setdefault(row["image"], {"manifest_record": row, "categories": []})
            item["categories"].append(category)
    return selected, {
        "method": "random.sample from lexically sorted candidate paths",
        "seed": seed,
        "requested_fn_cases": num_fn,
        "available_fn_cases": len(fn_candidates),
        "selected_fn_cases": len(selected_fn),
        "requested_fp_cases": num_fp,
        "available_fp_cases": len(fp_candidates),
        "selected_fp_cases": len(selected_fp),
    }


def _nonnegative_int(record, key):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid {key} in manifest row for {record.get('image')!r}.")
    return value


def match_predictions(predictions, ground_truth_boxes, iou_threshold=0.5):
    """Greedy confidence-ordered one-to-one IoU matching with diagnostic IoUs."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be greater than 0 and at most 1.")
    normalized = []
    for index, prediction in enumerate(predictions):
        bbox = prediction.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError(f"Prediction {index} has an invalid bbox.")
        x, y, width, height = map(float, bbox[:4])
        if not all(map(math.isfinite, (x, y, width, height))) or width <= 0 or height <= 0:
            raise ValueError(f"Prediction {index} has an invalid or nonpositive bbox.")
        confidence = prediction.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence):
                raise ValueError(f"Prediction {index} has a non-finite confidence score.")
        normalized.append({
            "detector_index": index,
            "bbox": [x, y, width, height],
            "box_xyxy": [x, y, x + width, y + height],
            "confidence": confidence,
            "_sort_confidence": confidence if confidence is not None else float("-inf"),
        })

    gt_boxes = [list(map(float, box)) for box in ground_truth_boxes]
    if any(len(box) != 4 for box in gt_boxes):
        raise ValueError("Ground-truth boxes must use [x1, y1, x2, y2].")
    all_ious = [
        [calculate_iou(prediction["box_xyxy"], gt_box) for gt_box in gt_boxes]
        for prediction in normalized
    ]
    order = sorted(
        range(len(normalized)),
        key=lambda index: normalized[index]["_sort_confidence"],
        reverse=True,
    )
    matched_gt = set()
    prediction_records = [None] * len(normalized)
    for rank, pred_index in enumerate(order, start=1):
        prediction = normalized[pred_index]
        ious = all_ious[pred_index]
        if ious:
            best_all_index = max(range(len(ious)), key=ious.__getitem__)
            best_all_iou = ious[best_all_index]
        else:
            best_all_index = None
            best_all_iou = 0.0

        best_available_index = None
        best_available_iou = 0.0
        for gt_index, iou in enumerate(ious):
            if gt_index in matched_gt:
                continue
            if iou > best_available_iou:
                best_available_iou = iou
                best_available_index = gt_index

        matched = (
            best_available_index is not None
            and best_available_iou >= iou_threshold
        )
        if matched:
            matched_gt.add(best_available_index)
        prediction_records[pred_index] = {
            "detector_index": pred_index,
            "confidence_rank": rank,
            "bbox_xywh": prediction["bbox"],
            "bbox_xyxy": prediction["box_xyxy"],
            "confidence": prediction["confidence"],
            "best_iou": best_all_iou,
            "best_ground_truth_index": best_all_index,
            "best_iou_at_or_above_threshold": bool(best_all_iou >= iou_threshold),
            "available_best_iou": best_available_iou,
            "matched_ground_truth_index": best_available_index if matched else None,
            "matches_iou_threshold": bool(matched),
            "status": "tp" if matched else "fp",
        }

    ground_truth_records = [
        {
            "ground_truth_index": index,
            "bbox_xyxy": box,
            "matched": index in matched_gt,
            "status": "matched" if index in matched_gt else "fn",
        }
        for index, box in enumerate(gt_boxes)
    ]
    tp = len(matched_gt)
    fp = len(normalized) - tp
    fn = len(gt_boxes) - tp
    return {
        "ground_truth_boxes": ground_truth_records,
        "predicted_boxes": prediction_records,
        "counts": {"true_positives": tp, "false_positives": fp, "false_negatives": fn},
    }


def build_diagnostic_record(
    image_path,
    source_manifest_record,
    ground_truth_boxes,
    predictions,
    iou_threshold,
    image_shape,
):
    """Build one serializable record from one targeted detector rerun."""
    matched = match_predictions(predictions, ground_truth_boxes, iou_threshold)
    return {
        "image": image_path,
        "status": "evaluated",
        "source_manifest_counts": {
            "true_positives": source_manifest_record.get("true_positives"),
            "false_positives": source_manifest_record.get("false_positives"),
            "false_negatives": source_manifest_record.get("false_negatives"),
            "predicted_faces": source_manifest_record.get("predicted_faces"),
        },
        "image_dimensions": {"height": int(image_shape[0]), "width": int(image_shape[1])},
        **matched,
    }


def draw_diagnostic(image, diagnostic_record):
    """Return a copy of the image with GT (green) and prediction (red) boxes."""
    annotated = image.copy()
    for gt in diagnostic_record["ground_truth_boxes"]:
        x1, y1, x2, y2 = map(int, gt["bbox_xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"GT {gt['ground_truth_index']} {gt['status'].upper()}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

    for prediction in diagnostic_record["predicted_boxes"]:
        x1, y1, x2, y2 = map(int, prediction["bbox_xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        confidence = prediction["confidence"]
        confidence_label = "conf n/a" if confidence is None else f"conf {confidence:.2f}"
        label = (
            f"P{prediction['detector_index']} {confidence_label} "
            f"IoU {prediction['best_iou']:.2f} {prediction['status'].upper()}"
        )
        cv2.putText(
            annotated, label, (x1, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    return annotated


def load_manifest(manifest_path):
    path = validate_manifest_input_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation manifest must contain a JSON object.")
    if manifest.get("dataset") != "WIDER FACE validation set":
        raise ValueError("Manifest is not a WIDER FACE validation-set evaluation.")
    image_root = manifest.get("image_root")
    if image_root:
        if not isinstance(image_root, str) or Path(image_root).resolve() != IMAGE_ROOT.resolve():
            raise ValueError("Manifest image_root does not match the local WIDER FACE dataset.")
    annotation_file = manifest.get("annotation_file")
    if annotation_file:
        if not isinstance(annotation_file, str) or Path(annotation_file).resolve() != ANNOTATION_FILE.resolve():
            raise ValueError("Manifest annotation_file does not match the local WIDER FACE dataset.")
    expected_model = manifest.get("model_name")
    if expected_model is not None and not isinstance(expected_model, str):
        raise ValueError("Manifest model_name must be a string.")
    if expected_model and expected_model != Path(DETECTOR_MODEL_PATH).name:
        raise ValueError(
            "Manifest detector model name does not match the currently configured FaceDetector: "
            f"{expected_model} != {Path(DETECTOR_MODEL_PATH).name}"
        )
    return path, manifest


def _valid_ground_truth_boxes(boxes):
    if not isinstance(boxes, list):
        return False
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        try:
            x1, y1, x2, y2 = map(float, box)
        except (TypeError, ValueError):
            return False
        if not all(map(math.isfinite, (x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
            return False
    return True


def _write_json_exclusive(path, data):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _safe_image_filename(image_path):
    digest = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:10]
    stem = Path(image_path).stem.replace(" ", "_")
    return f"{stem}_{digest}.jpg"


def run_diagnostics(
    manifest_path,
    num_fn=10,
    num_fp=10,
    seed=None,
    output_dir=None,
    detector_factory=FaceDetector,
):
    """Run the detector on selected manifest error images and save diagnostics."""
    manifest_path, manifest = load_manifest(manifest_path)
    config, config_sources = detector_configuration(manifest)
    selected, selection = select_error_cases(manifest, num_fn, num_fp, seed)
    if not selected:
        raise ValueError("Manifest contains no FN or FP cases matching the requested counts.")

    annotations = {}
    for image_path, boxes in load_annotations():
        if image_path in annotations:
            raise ValueError(f"Duplicate image path in WIDER FACE annotations: {image_path}")
        annotations[image_path] = boxes
    for image_path in selected:
        validate_image_path(image_path)
    out_dir = validate_diagnostic_output_dir(output_dir, manifest_path)
    detector = detector_factory(
        score_threshold=config["score_threshold"],
        nms_threshold=config["nms_threshold"],
        max_dimension=config["max_dimension"],
    )
    # Exclusive directory creation rejects a race or a pre-existing path.
    out_dir.mkdir(exist_ok=False)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=False)
    diagnostic_records = []
    aggregate = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    status_counts = {}

    for image_path, selection_info in selected.items():
        source_record = selection_info["manifest_record"]
        gt_boxes = annotations.get(image_path)
        if gt_boxes is None:
            record = {
                "image": image_path,
                "status": "invalid_annotation",
                "error": "Image path is missing from WIDER FACE annotations.",
                "categories": selection_info["categories"],
                "source_manifest_counts": source_record,
            }
        elif not _valid_ground_truth_boxes(gt_boxes):
            record = {
                "image": image_path,
                "status": "invalid_annotation",
                "error": "Ground-truth annotation contains an invalid box.",
                "categories": selection_info["categories"],
                "source_manifest_counts": source_record,
                "ground_truth_boxes": gt_boxes,
            }
        elif len(gt_boxes) != source_record.get("ground_truth_faces"):
            record = {
                "image": image_path,
                "status": "invalid_annotation",
                "error": "Ground-truth count differs from the source manifest.",
                "categories": selection_info["categories"],
                "source_manifest_counts": source_record,
                "ground_truth_boxes": gt_boxes,
            }
        else:
            image_file = validate_image_path(image_path)
            try:
                image = cv2.imread(str(image_file))
            except Exception as error:
                image = None
                image_read_error = str(error)
            else:
                image_read_error = "cv2.imread returned no image."
            if image is None:
                record = {
                    "image": image_path,
                    "status": "unreadable_image",
                    "error": image_read_error,
                    "categories": selection_info["categories"],
                    "source_manifest_counts": source_record,
                    "ground_truth_boxes": gt_boxes,
                }
            else:
                try:
                    predictions = detector.detect(image)
                    record = build_diagnostic_record(
                        image_path,
                        source_record,
                        gt_boxes,
                        predictions,
                        config["iou_threshold"],
                        image.shape,
                    )
                    record["categories"] = selection_info["categories"]
                    for key, value in record["counts"].items():
                        aggregate[key] += value
                    try:
                        annotated = draw_diagnostic(image, record)
                        filename = _safe_image_filename(image_path)
                        image_output = images_dir / filename
                        if not cv2.imwrite(str(image_output), annotated):
                            record["visualization_error"] = "cv2.imwrite returned false."
                        else:
                            record["visualization"] = str(image_output.relative_to(out_dir))
                    except Exception as error:
                        record["visualization_error"] = str(error)
                except Exception as error:
                    record = {
                        "image": image_path,
                        "status": "diagnostic_error",
                        "error": str(error),
                        "categories": selection_info["categories"],
                        "source_manifest_counts": source_record,
                        "ground_truth_boxes": gt_boxes,
                    }
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
        diagnostic_records.append(record)

    summary = {
        "evaluation_type": "targeted WIDER FACE error diagnostic rerun",
        "reproduction_note": (
            "Predictions in this report are targeted reruns on selected images. "
            "They are not the exact predictions from the source evaluation, which "
            "did not store prediction boxes."
        ),
        "source_manifest": str(manifest_path),
        "source_evaluation_counts": manifest.get("counts"),
        "source_evaluation_metrics": manifest.get("custom_metrics"),
        "dataset": "WIDER FACE validation set",
        "annotation_file": str(ANNOTATION_FILE),
        "image_root": str(IMAGE_ROOT),
        "model_name": Path(DETECTOR_MODEL_PATH).name,
        "model_name_source": (
            "manifest"
            if manifest.get("model_name")
            else "currently configured FaceDetector model"
        ),
        "detector_parameters": {
            key: config[key] for key in ("score_threshold", "nms_threshold", "max_dimension")
        },
        "parameter_sources": config_sources,
        "iou_threshold": config["iou_threshold"],
        "selection": selection,
        "selected_unique_images": len(selected),
        "status_counts": status_counts,
        "diagnostic_rerun_counts": aggregate,
        "images": diagnostic_records,
    }
    _write_json_exclusive(out_dir / "diagnostics.json", summary)
    print(f"Targeted diagnostic rerun complete: {len(selected)} unique images")
    print(f"Results: {out_dir / 'diagnostics.json'}")
    print(f"Visualizations: {images_dir}")
    print("These results are not an exact reproduction of the source evaluation.")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rerun selected WIDER FACE FN/FP images for visual diagnostics."
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=EVALUATION_OUTPUT_DIR / "widerface_single_face_seed42_all.json",
    )
    parser.add_argument("--num-fn", type=int, default=10)
    parser.add_argument("--num-fp", type=int, default=10)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Selection seed; defaults to the seed recorded in the source manifest.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="New direct-child directory under data/private/evaluation; must not exist.",
    )
    args = parser.parse_args(argv)
    if args.num_fn < 0 or args.num_fp < 0:
        parser.error("--num-fn and --num-fp must be nonnegative.")
    try:
        run_diagnostics(
            args.manifest,
            num_fn=args.num_fn,
            num_fp=args.num_fp,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
