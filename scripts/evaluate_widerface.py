"""Run a custom sampled WIDER FACE validation-set detector experiment.

This script reports custom precision/recall/F1 on an optional seeded sample.
It does not implement or report the official WIDER FACE AP benchmark metric.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_engine.detector import FaceDetector, MODEL_PATH as DETECTOR_MODEL_PATH


IMAGE_ROOT = (
    PROJECT_ROOT / "data/benchmarks/widerface/"
    "WIDER_val/WIDER_val/images"
)
ANNOTATION_FILE = (
    PROJECT_ROOT / "data/benchmarks/widerface/"
    "annotations/wider_face_split/wider_face_val_bbx_gt.txt"
)
DATASET_ROOT = PROJECT_ROOT / "data/benchmarks/widerface"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "data/private/evaluation"


def validate_image_path(image_path):
    relative_path = Path(image_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Annotation image path must stay relative to image root: {image_path}")

    root = IMAGE_ROOT.resolve()
    resolved = (IMAGE_ROOT / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Annotation image path escapes the image root: {image_path}"
        ) from error
    return resolved


def validate_manifest_path(manifest_path):
    output = Path(manifest_path).resolve()
    dataset_root = DATASET_ROOT.resolve()
    output_root = EVALUATION_OUTPUT_DIR.resolve()
    expected_output_root = Path(os.path.abspath(EVALUATION_OUTPUT_DIR))
    if output_root != expected_output_root:
        raise ValueError(
            "Evaluation output directory must not resolve through a symlink: "
            f"{EVALUATION_OUTPUT_DIR}"
        )
    if EVALUATION_OUTPUT_DIR.exists() and not EVALUATION_OUTPUT_DIR.is_dir():
        raise ValueError(f"Evaluation output path is not a directory: {EVALUATION_OUTPUT_DIR}")
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            "Evaluation manifest must be inside the dedicated directory "
            f"{EVALUATION_OUTPUT_DIR}: {output}"
        ) from error
    if output == output_root:
        raise ValueError(f"Manifest file path cannot be the evaluation directory: {output}")
    if output.exists() and output.is_dir():
        raise ValueError(f"Manifest path is a directory: {output}")
    if output.exists() and output.stat().st_nlink > 1:
        raise ValueError(f"Manifest path has multiple hard links and is unsafe: {output}")
    if output == ANNOTATION_FILE.resolve() or output == dataset_root or dataset_root in output.parents:
        raise ValueError(f"Manifest path must be outside the WIDER FACE dataset: {output}")
    return output


def load_annotations():
    lines = ANNOTATION_FILE.read_text(encoding="utf-8").splitlines()
    entries = []
    index = 0

    while index < len(lines):
        image_path = lines[index].strip()
        index += 1
        if not image_path:
            continue
        if index >= len(lines):
            raise ValueError(f"Missing face count for annotation: {image_path}")

        face_count = int(lines[index].strip())
        index += 1
        if face_count < 0:
            raise ValueError(f"Invalid negative face count for annotation: {image_path}")
        boxes = []
        for _ in range(face_count):
            if index >= len(lines):
                raise ValueError(f"Truncated boxes for annotation: {image_path}")
            values = list(map(float, lines[index].split()))
            index += 1
            if len(values) < 4:
                raise ValueError(f"Invalid bounding box row for {image_path}")
            x, y, width, height = values[:4]
            if width > 0 and height > 0:
                boxes.append([x, y, x + width, y + height])
        entries.append((image_path, boxes))

    return entries


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(
    limit=100,
    seed=42,
    iou_threshold=0.5,
    manifest_path=Path("data/private/widerface_custom_manifest.json"),
    score_threshold=0.5,
    nms_threshold=0.3,
    max_dimension=1280,
):
    if limit < 0:
        raise ValueError("--limit must be zero (all images) or a positive integer.")
    if not 0 < iou_threshold <= 1:
        raise ValueError("--iou must be greater than 0 and at most 1.")

    if not ANNOTATION_FILE.is_file():
        raise FileNotFoundError(f"WIDER FACE annotation file not found: {ANNOTATION_FILE}")
    if not IMAGE_ROOT.is_dir():
        raise NotADirectoryError(f"WIDER FACE image root not found: {IMAGE_ROOT}")
    manifest_path = validate_manifest_path(manifest_path)

    entries = load_annotations()
    if not entries:
        raise ValueError("WIDER FACE annotation file contains no image entries.")
    seen_paths = set()
    for image_path, _ in entries:
        validate_image_path(image_path)
        if image_path in seen_paths:
            raise ValueError(f"Duplicate image path in WIDER FACE annotations: {image_path}")
        seen_paths.add(image_path)
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    selected = shuffled[:limit] if limit else shuffled
    selected_paths = {image_path for image_path, _ in selected}
    not_sampled = [image_path for image_path, _ in entries if image_path not in selected_paths]

    detector = FaceDetector(
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        max_dimension=max_dimension,
    )

    tp = fp = fn = 0
    total_gt = total_predictions = 0
    sampled_gt = sum(len(boxes) for _, boxes in selected)
    unreadable = []
    sample_records = []

    for image_path, gt_boxes in selected:
        image_file = validate_image_path(image_path)
        image = cv2.imread(str(image_file))
        if image is None:
            unreadable.append(image_path)
            sample_records.append({
                "image": image_path,
                "status": "excluded",
                "reason": "unreadable_image",
                "ground_truth_faces": len(gt_boxes),
            })
            continue

        predictions = detector.detect(image)
        pred_boxes = [
            [x, y, x + width, y + height]
            for x, y, width, height in (item["bbox"] for item in predictions)
        ]
        order = sorted(
            range(len(predictions)),
            key=lambda position: predictions[position]["confidence"],
            reverse=True,
        )

        matched_gt = set()
        image_tp = image_fp = 0
        for pred_index in order:
            best_iou = 0.0
            best_gt_index = None
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched_gt:
                    continue
                score = calculate_iou(pred_boxes[pred_index], gt_box)
                if score > best_iou:
                    best_iou = score
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= iou_threshold:
                tp += 1
                image_tp += 1
                matched_gt.add(best_gt_index)
            else:
                fp += 1
                image_fp += 1

        image_fn = len(gt_boxes) - len(matched_gt)
        fn += image_fn
        total_gt += len(gt_boxes)
        total_predictions += len(predictions)
        sample_records.append({
            "image": image_path,
            "status": "evaluated",
            "ground_truth_faces": len(gt_boxes),
            "predicted_faces": len(predictions),
            "true_positives": image_tp,
            "false_positives": image_fp,
            "false_negatives": image_fn,
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    manifest = {
        "evaluation_type": "custom WIDER FACE validation-set sample",
        "official_benchmark_ap": False,
        "method_note": (
            "Greedy confidence-ordered one-to-one IoU matching with a fixed "
            "IoU cutoff; aggregate precision, recall, and F1 are custom sample "
            "metrics, not official WIDER FACE AP."
        ),
        "dataset": "WIDER FACE validation set",
        "annotation_file": str(ANNOTATION_FILE),
        "image_root": str(IMAGE_ROOT),
        "seed": seed,
        "requested_limit": limit,
        "sampled_image_count": len(selected),
        "iou_threshold": iou_threshold,
        "model_name": Path(DETECTOR_MODEL_PATH).name,
        "detector_parameters": {
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            "max_dimension": max_dimension,
        },
        "counts": {
            "total_dataset_images": len(entries),
            "sampled_images": len(selected),
            "evaluated_images": len(sample_records) - len(unreadable),
            "unreadable_images": len(unreadable),
            "sampled_ground_truth_faces": sampled_gt,
            "evaluated_ground_truth_faces": total_gt,
            "excluded_ground_truth_faces": sampled_gt - total_gt,
            "predictions": total_predictions,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "custom_metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "denominator_policy": (
            "Precision, recall, and F1 use readable sampled images only. "
            "Unreadable sampled images and their ground-truth boxes are excluded; "
            "their counts are reported separately."
        ),
        "sampled_images": sample_records,
        "excluded_samples": {
            "unreadable": unreadable,
            "not_sampled_due_to_limit": not_sampled,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nCUSTOM WIDER FACE VALIDATION SAMPLE")
    print("-----------------------------------")
    print("This is not official WIDER FACE benchmark AP.")
    print(f"Images in annotation set: {len(entries)}")
    print(f"Images sampled: {len(selected)}")
    print(f"Images evaluated: {len(sample_records) - len(unreadable)}")
    print(f"Unreadable images: {len(unreadable)}")
    print(f"Ground-truth faces: {total_gt}")
    print(f"Predictions: {total_predictions}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Custom precision: {precision:.4f}")
    print(f"Custom recall: {recall:.4f}")
    print(f"Custom F1: {f1:.4f}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a custom WIDER FACE sample detector evaluation (not official AP)."
    )
    parser.add_argument("--limit", type=int, default=100,
                        help="Seeded sample size; use 0 for all annotated images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--manifest", type=Path,
        default=EVALUATION_OUTPUT_DIR / "widerface_custom_manifest.json",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--max-dimension", type=int, default=1280)
    args = parser.parse_args()

    evaluate(
        limit=args.limit,
        seed=args.seed,
        iou_threshold=args.iou,
        manifest_path=args.manifest,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_dimension=args.max_dimension,
    )
