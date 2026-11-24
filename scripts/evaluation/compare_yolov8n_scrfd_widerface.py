"""Compare YOLOv8n-Face and SCRFD-10G on the complete WIDER FACE val set."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (str(SCRIPTS_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import numpy as np

from evaluation.common import calculate_detection_metrics
from evaluation.widerface_common import (
    EVALUATION_OUTPUT_DIR,
    IMAGE_ROOT,
    load_annotations,
    validate_image_path,
)
from evaluation import compare_widerface_detectors as shared
from evaluation.yolov8n_face_smoke import MODEL_PATH as YOLO_MODEL_PATH
from evaluation.yolov8n_face_smoke import YOLOv8FaceONNXAdapter


OUTPUT_DIR = EVALUATION_OUTPUT_DIR / "yolov8n_vs_scrfd_widerface_full3226_20261123"
SCRFD_MODEL_PATH = PROJECT_ROOT / "data/private/models/scrfd_10g_bnkps.onnx"
EXPECTED_IMAGES = 3226


def _new_counts():
    return {
        "images_attempted": 0,
        "images_evaluated": 0,
        "unreadable_images": 0,
        "unreadable_ground_truth_faces_excluded": 0,
        "detector_exceptions": 0,
        "predictions": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "failure_categories": {
            "image_read_failure": 0,
            "preprocessing_exception": 0,
            "detector_exception": 0,
            "prediction_validation_exception": 0,
            "matching_exception": 0,
        },
        "ground_truth_by_height": {name: 0 for name, _, _ in shared.FACE_HEIGHT_GROUPS},
        "matched_ground_truth_by_height": {name: 0 for name, _, _ in shared.FACE_HEIGHT_GROUPS},
        "false_negatives_by_height": {name: 0 for name, _, _ in shared.FACE_HEIGHT_GROUPS},
    }


def _add_failure_misses(counts, gt_boxes, category):
    counts["failure_categories"][category] += 1
    if category == "detector_exception":
        counts["detector_exceptions"] += 1
    counts["false_negatives"] += len(gt_boxes)
    for box in gt_boxes:
        group = shared._box_height_group(box)
        counts["ground_truth_by_height"][group] += 1
        counts["false_negatives_by_height"][group] += 1


def _normalize(predictions, scale):
    normalized = []
    for prediction in predictions:
        box = [float(value) / scale for value in prediction["bbox"]]
        confidence = float(prediction["confidence"])
        if len(box) != 4 or not np.isfinite(box).all() or not np.isfinite(confidence):
            raise RuntimeError("Detector returned invalid box or confidence values.")
        row = {"bbox": box, "confidence": confidence}
        if "landmarks" in prediction:
            points = [float(value) / scale for value in prediction["landmarks"]]
            if len(points) != 10 or not np.isfinite(points).all():
                raise RuntimeError("Detector returned malformed five-point landmarks.")
            row["landmarks"] = points
        normalized.append(row)
    return normalized


def _resource_sampler():
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process(os.getpid())
    process.cpu_percent(None)  # start the non-blocking measurement interval
    before = psutil.virtual_memory()
    return {
        "psutil": psutil,
        "process": process,
        "cpu_samples_percent": [],
        "rss_peak_bytes_sampled": process.memory_info().rss,
        "system_memory_total_bytes": before.total,
        "system_memory_available_start_bytes": before.available,
    }


def _sample_resources(sampler):
    if sampler is None:
        return
    process = sampler["process"]
    sampler["cpu_samples_percent"].append(process.cpu_percent(None))
    sampler["rss_peak_bytes_sampled"] = max(
        sampler["rss_peak_bytes_sampled"], process.memory_info().rss
    )


def _latency_summary(samples):
    return shared._latency_summary(samples)


def _make_report(detectors, image_count, benchmark_seconds, resources, init_seconds):
    summaries = {}
    for name, result in detectors.items():
        counts = result["counts"]
        metrics = calculate_detection_metrics(
            counts["true_positives"], counts["false_positives"], counts["false_negatives"]
        )
        summaries[name] = {
            "label": result["label"],
            "model": str(result["model"]),
            "thresholds": result["thresholds"],
            "counts": counts,
            "custom_metrics_iou_0_5": metrics,
            "recall_by_face_height": {
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
                for group, _, _ in shared.FACE_HEIGHT_GROUPS
            },
            "inference_latency_ms": _latency_summary(result["latency_samples_ms"]),
            "total_inference_time_s": sum(result["latency_samples_ms"]) / 1000.0,
            "prediction_file": f"predictions/{name}.jsonl",
        }
    if resources is None:
        resource_report = {"available": False, "note": "psutil is not installed in the existing environment."}
    else:
        psutil = resources["psutil"]
        cpu = resources["cpu_samples_percent"]
        after = psutil.virtual_memory()
        resource_report = {
            "available": True,
            "logical_cpu_count": os.cpu_count(),
            "process_cpu_percent_sample_mean": statistics.fmean(cpu) if cpu else None,
            "process_cpu_percent_sample_max": max(cpu) if cpu else None,
            "process_rss_peak_sampled_bytes": resources["rss_peak_bytes_sampled"],
            "system_memory_total_bytes": resources["system_memory_total_bytes"],
            "system_memory_available_start_bytes": resources["system_memory_available_start_bytes"],
            "system_memory_available_end_bytes": after.available,
            "sampling": "Non-blocking process CPU and RSS samples every 25 images; RSS peak is sampled, not OS high-water mark.",
        }
    return {
        "evaluation_type": "custom WIDER FACE complete validation-set detector comparison",
        "official_benchmark_ap": False,
        "metric_note": "All precision/recall/F1 values use custom confidence-ordered greedy one-to-one matching at IoU >= 0.5. These are NOT official WIDER FACE Easy/Medium/Hard AP.",
        "dataset": {
            "name": "WIDER FACE validation set",
            "image_root": str(IMAGE_ROOT),
            "annotation_file": str(shared.ANNOTATION_FILE),
            "images_in_annotation_file": image_count,
            "same_images_and_ground_truth_for_both_detectors": True,
        },
        "protocol": {
            "iou_threshold": shared.IOU_THRESHOLD,
            "common_preprocessing": "Existing comparison preprocessing: aspect-preserving resize capped at 1280, then bottom/right pad to stride 32; same prepared image passed to both adapters.",
            "latency_scope": "Shared common resize/padding plus detector adapter preprocessing, CPU inference, decoding, and NMS; excludes disk image loading, matching, and report output.",
            "unreadable_images": "Excluded from both detectors and their ground-truth denominator; reported in counts.",
            "detector_failures": "Readable-image failures count all ground-truth faces as false negatives.",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
            "providers": {"yolov8n_face": ["CPUExecutionProvider"], "scrfd_10g_kps": ["CPUExecutionProvider"]},
        },
        "timing": {
            "benchmark_total_execution_time_s": benchmark_seconds,
            "model_initialization_time_s": init_seconds,
            "includes": "annotation loading, model initialization, all image inference and metric calculation, and streamed prediction-file writing; excludes final Markdown/JSON report serialization.",
        },
        "resources": resource_report,
        "licensing": {
            "yolov8_model_metadata_license": "AGPL-3.0",
            "yolov8_inference_repository_license": "No LICENSE file; licensing remains unresolved.",
            "note": "This is an internal research comparison. Confirm applicable AGPL/model and upstream code terms before redistribution or product use.",
        },
        "detectors": summaries,
    }


def _markdown(report):
    lines = [
        "# YOLOv8n-Face vs SCRFD-10G — WIDER FACE validation",
        "",
        f"Images evaluated: **{report['dataset']['images_in_annotation_file']}**",
        "",
        "**Metric scope:** custom confidence-ordered greedy matching at IoU 0.5. These precision, recall, and F1 values are **not official WIDER FACE Easy/Medium/Hard AP**.",
        "",
        "## Results",
        "",
        "| Detector | TP | FP | FN | Precision | Recall | F1 | Mean ms | Median ms | P95 ms | Total inference s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["detectors"].items():
        c, m, l = result["counts"], result["custom_metrics_iou_0_5"], result["inference_latency_ms"]
        lines.append(
            f"| {result['label']} | {c['true_positives']} | {c['false_positives']} | {c['false_negatives']} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{l['mean_ms']:.2f} | {l['median_ms']:.2f} | {l['p95_ms']:.2f} | {result['total_inference_time_s']:.2f} |"
        )
    lines += ["", "## Recall by ground-truth face height", "", "| Detector | <=20 px | >20 to 40 px | >40 px |", "|---|---:|---:|---:|"]
    for result in report["detectors"].values():
        groups = result["recall_by_face_height"]
        lines.append(
            f"| {result['label']} | {groups['le_20_px']['recall']:.4f} "
            f"({groups['le_20_px']['true_positives']}/{groups['le_20_px']['ground_truth_faces']}) | "
            f"{groups['gt_20_to_40_px']['recall']:.4f} "
            f"({groups['gt_20_to_40_px']['true_positives']}/{groups['gt_20_to_40_px']['ground_truth_faces']}) | "
            f"{groups['gt_40_px']['recall']:.4f} "
            f"({groups['gt_40_px']['true_positives']}/{groups['gt_40_px']['ground_truth_faces']}) |"
        )
    lines += [
        "",
        "## Runtime and resources",
        "",
        f"- Total benchmark execution: {report['timing']['benchmark_total_execution_time_s']:.2f} s.",
        f"- Model initialization: {report['timing']['model_initialization_time_s']:.2f} s.",
        f"- CPU/memory sampling: `{json.dumps(report['resources'], ensure_ascii=False)}`",
        "- Inference used CPU providers. Prediction files are under `predictions/`.",
        "",
        "## Configuration and limitations",
        "",
        "- Same 3,226 annotated validation images, common resize/padding, CPU execution, and custom IoU=0.5 matching for both detectors.",
        "- YOLOv8n-Face score/NMS thresholds: 0.50/0.45; SCRFD-10G-KPS: 0.50/0.40 (the existing adapter defaults).",
        "- YOLOv8 ONNX metadata declares **AGPL-3.0**. The cited inference repository has no LICENSE file, so its code license remains unresolved. Confirm terms before redistribution or product use.",
        "- Official WIDER FACE AP was not calculated; MATLAB/Octave is unavailable.",
        "",
    ]
    return "\n".join(lines)


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to reuse existing experiment directory: {OUTPUT_DIR}")
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLOv8n-Face model is missing: {YOLO_MODEL_PATH}")
    if not SCRFD_MODEL_PATH.is_file():
        raise FileNotFoundError(f"SCRFD KPS model is missing: {SCRFD_MODEL_PATH}")

    benchmark_started = time.perf_counter()
    annotations = load_annotations()
    if len(annotations) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} annotated validation images, got {len(annotations)}.")

    init_started = time.perf_counter()
    detectors = {
        "yolov8n_face": {
            "adapter": YOLOv8FaceONNXAdapter(YOLO_MODEL_PATH),
            "label": "YOLOv8n-Face",
            "model": YOLO_MODEL_PATH,
            "thresholds": {"confidence": 0.5, "nms_iou": 0.45},
            "counts": _new_counts(),
            "latency_samples_ms": [],
        },
        "scrfd_10g_kps": {
            "adapter": shared.create_scrfd_adapter(SCRFD_MODEL_PATH),
            "label": "SCRFD-10G-KPS",
            "model": SCRFD_MODEL_PATH,
            "thresholds": {"confidence": 0.5, "nms_iou": 0.4},
            "counts": _new_counts(),
            "latency_samples_ms": [],
        },
    }
    init_seconds = time.perf_counter() - init_started
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    prediction_dir = OUTPUT_DIR / "predictions"
    prediction_dir.mkdir()
    resource_sampler = _resource_sampler()

    with (prediction_dir / "yolov8n_face.jsonl").open("x", encoding="utf-8") as yolo_file, \
         (prediction_dir / "scrfd_10g_kps.jsonl").open("x", encoding="utf-8") as scrfd_file:
        output_files = {"yolov8n_face": yolo_file, "scrfd_10g_kps": scrfd_file}
        for image_index, (image_rel, gt_boxes) in enumerate(annotations, start=1):
            image_file = validate_image_path(image_rel)
            image, read_error = shared._read_evaluation_image(image_file)
            for detector in detectors.values():
                detector["counts"]["images_attempted"] += 1
            if read_error:
                for name, detector in detectors.items():
                    counts = detector["counts"]
                    counts["unreadable_images"] += 1
                    counts["unreadable_ground_truth_faces_excluded"] += len(gt_boxes)
                    counts["failure_categories"]["image_read_failure"] += 1
                    output_files[name].write(json.dumps({
                        "image": image_rel, "status": "excluded", "error": read_error,
                        "ground_truth_faces": len(gt_boxes), "detections": [],
                    }) + "\n")
                continue

            preprocess_started = time.perf_counter()
            try:
                prepared, scale = shared.resize_preserving_aspect(image)
                shared_preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
            except Exception as error:
                shared_preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
                for name, detector in detectors.items():
                    counts = detector["counts"]
                    _add_failure_misses(counts, gt_boxes, "preprocessing_exception")
                    detector["latency_samples_ms"].append(shared_preprocess_ms)
                    output_files[name].write(json.dumps({
                        "image": image_rel, "status": "failed", "error": str(error),
                        "ground_truth_faces": len(gt_boxes), "detections": [],
                    }) + "\n")
                continue

            for name, detector in detectors.items():
                counts = detector["counts"]
                infer_started = time.perf_counter()
                failure_category = "detector_exception"
                try:
                    raw = detector["adapter"].detect(prepared)
                    failure_category = "prediction_validation_exception"
                    normalized = _normalize(raw, scale)
                    latency_ms = shared_preprocess_ms + (time.perf_counter() - infer_started) * 1000.0
                    failure_category = "matching_exception"
                    matches = shared.match_image_predictions(gt_boxes, normalized)
                except Exception as error:
                    latency_ms = shared_preprocess_ms + (time.perf_counter() - infer_started) * 1000.0
                    _add_failure_misses(counts, gt_boxes, failure_category)
                    detector["latency_samples_ms"].append(latency_ms)
                    output_files[name].write(json.dumps({
                        "image": image_rel, "status": "failed", "error": str(error),
                        "ground_truth_faces": len(gt_boxes), "detections": [],
                    }) + "\n")
                    continue

                counts["images_evaluated"] += 1
                counts["predictions"] += len(normalized)
                for key in ("true_positives", "false_positives", "false_negatives"):
                    counts[key] += matches[key]
                for key in ("ground_truth_by_height", "matched_ground_truth_by_height", "false_negatives_by_height"):
                    for group, value in matches[key].items():
                        counts[key][group] += value
                detector["latency_samples_ms"].append(latency_ms)
                output_files[name].write(json.dumps({
                    "image": image_rel, "status": "evaluated",
                    "ground_truth_faces": len(gt_boxes), "detections": normalized,
                }, separators=(",", ":")) + "\n")

            if image_index % 25 == 0 or image_index == len(annotations):
                _sample_resources(resource_sampler)
                elapsed = time.perf_counter() - benchmark_started
                print(f"Progress: {image_index}/{len(annotations)} images; {elapsed:.1f}s elapsed", file=sys.stderr, flush=True)

    benchmark_seconds = time.perf_counter() - benchmark_started
    report = _make_report(detectors, len(annotations), benchmark_seconds, resource_sampler, init_seconds)
    report_path = OUTPUT_DIR / "comparison_results.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    markdown_path = OUTPUT_DIR / "comparison_report.md"
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(_markdown(report))
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "benchmark_total_execution_time_s": benchmark_seconds,
        "detectors": {
            name: {"counts": value["counts"], "metrics": calculate_detection_metrics(
                value["counts"]["true_positives"], value["counts"]["false_positives"], value["counts"]["false_negatives"]
            ), "latency_ms": _latency_summary(value["latency_samples_ms"])}
            for name, value in detectors.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
