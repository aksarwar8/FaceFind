"""Evaluate FaceFind open-set identification using an LFW-derived gallery.

Requires the official LFW image tree under data/benchmarks/lfw/lfw_funneled
unless --images-dir is supplied. All reports are exclusive-created below
data/private/evaluation/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for _path in (str(SCRIPTS_ROOT), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation.common import validate_output_path, write_manifest_exclusive
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.gallery import build_gallery
from face_engine.match import FaceMatcher
from face_engine.pipeline import recognize_image

DATA_ROOT = PROJECT_ROOT / "data" / "benchmarks" / "lfw"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "private" / "evaluation"
SUPPORTED = {".jpg", ".jpeg", ".png"}
SEED = 42
KNOWN_IDENTITY_COUNT = 200
UNKNOWN_IDENTITY_COUNT = 100
CONFIGURATIONS = {
    "production": {"recognition_threshold": 0.50, "possible_threshold": 0.35,
                   "margin_threshold": 0.12, "possible_margin_threshold": 0.087,
                   "rationale": "Exact FaceMatcher production defaults."},
    "lower_possible_margin": {"recognition_threshold": 0.50, "possible_threshold": 0.35,
                              "margin_threshold": 0.12, "possible_margin_threshold": 0.05,
                              "rationale": "Isolates a less restrictive possible-match margin."},
    "higher_possible_margin": {"recognition_threshold": 0.50, "possible_threshold": 0.35,
                               "margin_threshold": 0.12, "possible_margin_threshold": 0.15,
                               "rationale": "Isolates a more restrictive possible-match margin."},
}


def discover_eligible_identities(image_root):
    root = Path(image_root).resolve()
    identities = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        paths = sorted(p.resolve() for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in SUPPORTED)
        if len(paths) >= 2:
            identities[folder.name] = paths
    return identities


def _usable_detection(detection):
    """Require a finite, positive-size YuNet box and five finite landmarks."""
    try:
        bbox = detection["bbox"]
        landmarks = detection["landmarks"]
        confidence = float(detection["confidence"])
        values = [float(value) for value in bbox + landmarks]
        return (
            len(bbox) == 4
            and len(landmarks) == 10
            and all(math.isfinite(value) for value in values)
            and float(bbox[2]) > 0
            and float(bbox[3]) > 0
            and math.isfinite(confidence)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def preflight_gallery_reference(identity, image_path, detector, embedder):
    """Accept a gallery reference only when one usable face and embedding exist."""
    image_path = Path(image_path)
    record = {"identity": identity, "reference_image": str(image_path)}
    try:
        image = cv2.imread(str(image_path))
    except Exception as error:
        return {**record, "accepted": False, "reason": "gallery_image_read_error",
                "detail": str(error)}
    if image is None:
        return {**record, "accepted": False, "reason": "unreadable_gallery_image",
                "face_count": 0}
    try:
        faces = detector.detect(image)
    except Exception as error:
        return {**record, "accepted": False, "reason": "gallery_detection_error",
                "detail": str(error)}

    record["face_count"] = len(faces)
    if len(faces) != 1:
        return {**record, "accepted": False,
                "reason": "no_face_detected" if not faces else "multiple_faces_detected"}
    detection = faces[0]
    if not _usable_detection(detection):
        return {**record, "accepted": False, "reason": "invalid_face_detection"}
    try:
        embedding = np.asarray(embedder.get_embedding(image, detection)).reshape(-1)
        if embedding.size == 0 or not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
            raise ValueError("Embedding is empty, non-finite, or zero length")
    except Exception as error:
        return {**record, "accepted": False, "reason": "gallery_embedding_error",
                "detail": str(error)}

    return {
        **record,
        "accepted": True,
        "reason": None,
        "detection": {
            "bbox": list(detection["bbox"]),
            "confidence": float(detection["confidence"]),
        },
    }


def make_split(identities, preflight_reference, seed=SEED,
               known_count=KNOWN_IDENTITY_COUNT,
               unknown_count=UNKNOWN_IDENTITY_COUNT):
    """Preflight known references, using a disjoint seeded reserve if needed.

    The first ``known_count`` identities are primary known candidates, the next
    ``unknown_count`` remain the unknown set, and later identities form the
    known-gallery reserve. Failed candidates are excluded rather than reused as
    unknowns. Only identities with accepted references contribute probes.
    """
    eligible_names = sorted(identities)
    if len(eligible_names) < known_count + unknown_count:
        raise ValueError(f"Need at least {known_count + unknown_count} eligible identities; found {len(eligible_names)}.")
    rng = random.Random(seed)
    rng.shuffle(eligible_names)

    primary_names = eligible_names[:known_count]
    unknown_names = eligible_names[known_count:known_count + unknown_count]
    reserve_names = eligible_names[known_count + unknown_count:]
    known = {}
    excluded = []
    replacements = []
    preflight_records = []
    replacement_targets = []

    for origin, candidates in (("primary", primary_names), ("reserve", reserve_names)):
        for identity in candidates:
            if len(known) >= known_count:
                break
            paths = list(identities[identity])
            rng.shuffle(paths)
            reference = paths[0]
            check = preflight_reference(identity, reference)
            check = {**check, "origin": origin}
            preflight_records.append(check)
            if not check.get("accepted", False):
                excluded.append(check)
                if origin == "primary":
                    replacement_targets.append(identity)
                continue

            known[identity] = {"gallery": reference, "probes": paths[1:]}
            if origin == "reserve":
                replaced = replacement_targets.pop(0) if replacement_targets else None
                replacements.append({
                    "identity": identity,
                    "reference_image": str(reference),
                    "replaces_identity": replaced,
                })
        if len(known) >= known_count:
            break

    if len(known) != known_count:
        raise ValueError(
            f"Only {len(known)} known identities had usable gallery references; "
            f"{known_count} required after checking {len(preflight_records)} candidates."
        )

    unknown = {identity: list(identities[identity]) for identity in unknown_names}
    if set(known).intersection(unknown):
        raise AssertionError("Known and unknown identity splits overlap.")
    return known, unknown, {
        "primary_candidates": primary_names,
        "unknown_identities": unknown_names,
        "reserve_replacements": replacements,
        "excluded_identities": excluded,
        "preflight_records": preflight_records,
    }


def classify_query(image_path, gallery, detector, embedder, matcher):
    image = cv2.imread(str(image_path))
    if image is None:
        return {"status": "failed", "failure": "unreadable_image"}
    try:
        result = recognize_image(image, gallery, detector, embedder, matcher)
    except Exception as error:
        return {"status": "failed", "failure": "pipeline_error", "detail": str(error)}
    if result["face_count"] == 0:
        return {"status": "failed", "failure": "no_face_detected"}
    if result["face_count"] != 1:
        return {"status": "failed", "failure": "multiple_faces_detected",
                "face_count": result["face_count"]}
    prediction = result["results"][0]
    if prediction["status"] == "error":
        return {"status": "failed", "failure": "embedding_or_match_error",
                "detail": prediction.get("error")}
    return {"status": "evaluated", "identity": prediction["identity"],
            "match_status": prediction["status"], "best_score": prediction["best_score"],
            "second_score": prediction["second_score"], "margin": prediction["margin"]}


def score_counts(rows, known):
    valid = [row for row in rows if row["result"]["status"] == "evaluated"]
    if known:
        correct = sum(r["result"]["identity"] == r["true_identity"] and
                      r["result"]["match_status"] in {"recognised", "possible"} for r in valid)
        incorrect = sum(r["result"]["identity"] != r["true_identity"] and
                        r["result"]["match_status"] in {"recognised", "possible"} for r in valid)
        rejected = sum(r["result"]["match_status"] == "unknown" for r in valid)
        possible = sum(r["result"]["match_status"] == "possible" for r in valid)
        recognized = sum(r["result"]["match_status"] == "recognised" for r in valid)
        return {"probes_total": len(rows), "probes_evaluated": len(valid),
                "failed_probes": len(rows) - len(valid), "correct_identification": correct,
                "incorrect_identification": incorrect, "rejected": rejected,
                "possible_classification": possible, "recognised_classification": recognized,
                "correct_identification_rate": correct / len(valid) if valid else 0.0,
                "incorrect_identification_rate": incorrect / len(valid) if valid else 0.0,
                "rejection_rate": rejected / len(valid) if valid else 0.0}
    false_id = sum(r["result"]["match_status"] in {"recognised", "possible"} for r in valid)
    correct_reject = sum(r["result"]["match_status"] == "unknown" for r in valid)
    return {"probes_total": len(rows), "probes_evaluated": len(valid),
            "failed_probes": len(rows) - len(valid), "false_identification": false_id,
            "correct_unknown_rejection": correct_reject,
            "false_identification_rate": false_id / len(valid) if valid else 0.0,
            "correct_unknown_rejection_rate": correct_reject / len(valid) if valid else 0.0,
            "possible_classification": sum(r["result"]["match_status"] == "possible" for r in valid)}


def evaluate(images_dir, output_dir, seed=SEED, known_count=KNOWN_IDENTITY_COUNT,
             unknown_count=UNKNOWN_IDENTITY_COUNT):
    output_dir = Path(output_dir)
    safe_report = validate_output_path(
        output_dir / "open_set_report.json",
        OUTPUT_ROOT,
        protected_roots=(DATA_ROOT, images_dir),
    )
    output_dir = safe_report.parent
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing evaluation directory: {output_dir}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_output_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=OUTPUT_ROOT
    ))
    published = False
    try:
        report = _evaluate_into_directory(
            images_dir, temporary_output_dir, seed, known_count, unknown_count
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Recheck containment immediately before publication in case a parent
        # path changed while the evaluation was running.
        safe_report = validate_output_path(
            output_dir / "open_set_report.json",
            OUTPUT_ROOT,
            protected_roots=(DATA_ROOT, images_dir),
        )
        output_dir = safe_report.parent
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse existing evaluation directory: {output_dir}"
            )
        os.rename(temporary_output_dir, output_dir)
        published = True
    finally:
        if not published and temporary_output_dir.exists():
            shutil.rmtree(temporary_output_dir)

    print(json.dumps({"output_dir": str(output_dir), "dataset": report["dataset"],
                      "threshold_comparison": report["threshold_comparison"]}, indent=2))
    return report


def _evaluate_into_directory(images_dir, output_dir, seed, known_count, unknown_count):
    """Build a complete run inside a private, not-yet-published directory."""
    identities = discover_eligible_identities(images_dir)
    detector, embedder = FaceDetector(), FaceEmbedder()
    known, unknown, split_audit = make_split(
        identities,
        lambda identity, path: preflight_gallery_reference(
            identity, path, detector, embedder
        ),
        seed,
        known_count,
        unknown_count,
    )
    staging = output_dir / "gallery_staging"
    staging.mkdir()
    try:
        for identity, split in known.items():
            shutil.copy2(split["gallery"], staging / f"{identity}{split['gallery'].suffix.lower()}")
        gallery = build_gallery(staging, detector, embedder)
    finally:
        shutil.rmtree(staging)

    gallery_identities = set(gallery["names"])
    if gallery_identities != set(known):
        missing = sorted(set(known) - gallery_identities)
        unexpected = sorted(gallery_identities - set(known))
        raise RuntimeError(
            "The production gallery builder did not reproduce the preflighted "
            f"identity set (missing={missing}, unexpected={unexpected}, "
            f"errors={gallery['errors']}, pending={gallery['pending']})."
        )
    if len(gallery_identities) != known_count:
        raise RuntimeError(
            f"Expected {known_count} gallery identities after preflight; "
            f"built {len(gallery_identities)}."
        )
    if gallery_identities.intersection(unknown):
        raise AssertionError("An unknown identity was included in the gallery.")

    probes = []
    for identity, split in known.items():
        for path in split["probes"]:
            probes.append({"split": "known", "true_identity": identity, "image": str(path)})
    for identity, paths in unknown.items():
        for path in paths:
            probes.append({"split": "unknown", "true_identity": identity, "image": str(path)})

    per_config = {}
    failures = Counter()
    for config_name, parameters in CONFIGURATIONS.items():
        matcher = FaceMatcher(**{key: value for key, value in parameters.items()
                                 if key in {"recognition_threshold", "possible_threshold",
                                            "margin_threshold", "possible_margin_threshold"}})
        known_rows, unknown_rows, per_identity = [], [], defaultdict(list)
        for probe in probes:
            result = classify_query(probe["image"], gallery, detector, embedder, matcher)
            row = {**probe, "true_identity_in_gallery": probe["true_identity"] in gallery_identities,
                   "result": result}
            if result["status"] == "failed": failures[result["failure"]] += 1
            if probe["split"] == "known":
                known_rows.append(row)
                per_identity[probe["true_identity"]].append(row)
            else:
                unknown_rows.append(row)
        per_config[config_name] = {
            "parameters": parameters,
            "known": score_counts(known_rows, True),
            "unknown": score_counts(unknown_rows, False),
            "per_identity_known": {identity: score_counts(rows, True)
                                   for identity, rows in sorted(per_identity.items())},
            "per_probe_results": known_rows + unknown_rows,
        }

    report = {
        "evaluation": "LFW-derived open-set identification via FaceFind production pipeline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"image_root": str(Path(images_dir).resolve()),
                    "eligible_identity_count": len(identities), "gallery_size": len(gallery["names"]),
                    "known_identity_count_selected": len(known),
                    "known_identity_count_in_gallery": len(gallery_identities),
                    "unknown_identity_count": len(unknown),
                    "excluded_identity_count": len(split_audit["excluded_identities"]),
                    "reserve_replacement_count": len(split_audit["reserve_replacements"]),
                    "gallery_build_errors": gallery["errors"], "gallery_pending": gallery["pending"],
                    "failed_image_loads": sum(1 for row in per_config["production"]["per_probe_results"]
                                               if row["result"].get("failure") == "unreadable_image"),
                    "failed_face_detections": sum(1 for row in per_config["production"]["per_probe_results"]
                                                  if row["result"].get("failure") in {"no_face_detected", "multiple_faces_detected"}),
                    "embedding_failures": sum(1 for row in per_config["production"]["per_probe_results"]
                                              if row["result"].get("failure") == "embedding_or_match_error")},
        "detector_parameters": {"score_threshold": 0.5, "nms_threshold": 0.3,
                                 "max_dimension": 1280},
        "models": {"detector": "face_detection_yunet_2023mar.onnx",
                   "embedder": "face_recognition_sface_2021dec.onnx"},
        "threshold_comparison": {name: {"parameters": value["parameters"],
                                         "known": value["known"], "unknown": value["unknown"]}
                                 for name, value in per_config.items()},
        "configurations": per_config,
        "reproducibility_manifest": {
            "seed": seed,
            "gallery_identities": sorted(gallery_identities),
            "selected_reference_paths": {
                identity: str(split["gallery"])
                for identity, split in sorted(known.items())
            },
            "known": {identity: {"gallery_image": str(value["gallery"]),
                                 "probe_images": [str(p) for p in value["probes"]]}
                      for identity, value in known.items()},
            "unknown": {identity: [str(p) for p in paths] for identity, paths in unknown.items()},
            "excluded_identities": split_audit["excluded_identities"],
            "reserve_replacements": split_audit["reserve_replacements"],
            "gallery_preflight_records": split_audit["preflight_records"],
            "detection_failures": {
                "gallery_reference_preflight": split_audit["excluded_identities"],
                "probe_failures": [
                    {"split": row["split"], "true_identity": row["true_identity"],
                     "image": row["image"], **row["result"]}
                    for row in per_config["production"]["per_probe_results"]
                    if row["result"].get("failure") in {
                        "no_face_detected", "multiple_faces_detected"
                    }
                ],
            },
            "probe_splits": {
                "known": {identity: [str(path) for path in value["probes"]]
                          for identity, value in known.items()},
                "unknown": {identity: [str(path) for path in paths]
                            for identity, paths in unknown.items()},
            },
            "gallery_builder": "face_engine.gallery.build_gallery",
            "query_pipeline": "face_engine.pipeline.recognize_image",
            "identity_overlap": sorted(set(known).intersection(unknown)),
            "note": (
                "Known identities are included only after reference preflight and "
                "must match the production gallery builder's actual identity list. "
                "Unknown identities are disjoint from known and gallery identities."
            ),
        },
        "failure_counts_by_reason_across_configurations": dict(failures),
    }
    write_manifest_exclusive(output_dir / "open_set_report.json", report)
    write_manifest_exclusive(output_dir / "reproducibility_manifest.json", report["reproducibility_manifest"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DATA_ROOT / "lfw_funneled")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--known-identities", type=int, default=KNOWN_IDENTITY_COUNT)
    parser.add_argument("--unknown-identities", type=int, default=UNKNOWN_IDENTITY_COUNT)
    args = parser.parse_args(argv)
    if not args.images_dir.is_dir(): parser.error(f"LFW image directory not found: {args.images_dir}")
    if args.known_identities < 150 or args.known_identities > 300:
        parser.error("--known-identities must be between 150 and 300.")
    if args.unknown_identities < 100: parser.error("--unknown-identities must be at least 100.")
    if args.output_dir is None:
        args.output_dir = OUTPUT_ROOT / f"open_set_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    safe_report = validate_output_path(args.output_dir / "open_set_report.json", OUTPUT_ROOT,
                                       protected_roots=(DATA_ROOT, args.images_dir))
    evaluate(args.images_dir, safe_report.parent, args.seed,
             args.known_identities, args.unknown_identities)


if __name__ == "__main__":
    main()
