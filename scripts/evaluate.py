"""Evaluate FaceFind recognition on an identity-organized query set.

Expected layout::

    reference_dir/<identity>.<image extension>
    evaluation_dir/known/<gallery identity>/<query images>
    evaluation_dir/unknown/<non-gallery identity>/<query images>

The manifest records this preassigned split and a seeded, deterministic query
order. This script does not tune recognition thresholds or produce group-photo
results; it uses the production recognition pipeline and matcher defaults.
SHA-256 checks identify byte-identical images only; resized, compressed, or
re-encoded copies are not detected.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from face_engine.detector import FaceDetector, MODEL_PATH as DETECTOR_MODEL_PATH
from face_engine.embedder import FaceEmbedder, MODEL_PATH as EMBEDDER_MODEL_PATH
from face_engine.gallery import SUPPORTED_EXTENSIONS, build_gallery
from face_engine.match import FaceMatcher
from face_engine.pipeline import recognize_image

EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "data/private/evaluation"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ensure_within(path, root, description):
    resolved_root = Path(root).resolve(strict=True)
    resolved_path = Path(path).resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{description} escapes its dataset root: {path}") from error
    return resolved_path


def validate_output_paths(output_csv, manifest_path, input_roots):
    outputs = [Path(output_csv).resolve(), Path(manifest_path).resolve()]
    if outputs[0] == outputs[1]:
        raise ValueError("CSV output and manifest output must be different paths.")

    output_root = EVALUATION_OUTPUT_DIR.resolve()
    expected_output_root = Path(os.path.abspath(EVALUATION_OUTPUT_DIR))
    if output_root != expected_output_root:
        raise ValueError(
            "Evaluation output directory must not resolve through a symlink: "
            f"{EVALUATION_OUTPUT_DIR}"
        )
    if EVALUATION_OUTPUT_DIR.exists() and not EVALUATION_OUTPUT_DIR.is_dir():
        raise ValueError(f"Evaluation output path is not a directory: {EVALUATION_OUTPUT_DIR}")

    roots = [Path(root).resolve() for root in input_roots]
    for output in outputs:
        if output == output_root:
            raise ValueError(f"Output file path cannot be the evaluation directory: {output}")
        if output.exists() and output.is_dir():
            raise ValueError(f"Output path is a directory: {output}")
        try:
            output.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                "Evaluation outputs must be inside the dedicated directory "
                f"{EVALUATION_OUTPUT_DIR}: {output}"
            ) from error
        if output.exists() and output.stat().st_nlink > 1:
            raise ValueError(f"Output path has multiple hard links and is unsafe: {output}")
        if any(output == root or root in output.parents for root in roots):
            raise ValueError(
                f"Output path must be outside all input directories: {output}"
            )
    return outputs


def validate_dataset(reference_dir, evaluation_dir):
    reference_dir = Path(reference_dir).resolve()
    evaluation_dir = Path(evaluation_dir).resolve()

    if not reference_dir.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {reference_dir}")
    if not evaluation_dir.is_dir():
        raise NotADirectoryError(f"Evaluation directory not found: {evaluation_dir}")

    references = image_files(reference_dir)
    for path in references:
        ensure_within(path, reference_dir, "Reference image")
    if not references:
        raise ValueError("Reference directory contains no supported image files.")

    reference_identities = {}
    for path in references:
        identity = path.stem.strip()
        key = identity.casefold()
        if not identity:
            raise ValueError(f"Reference image has an empty identity name: {path}")
        if key in reference_identities:
            raise ValueError(
                "Duplicate reference identity names (case-insensitive): "
                f"{reference_identities[key].name}, {path.name}"
            )
        reference_identities[key] = path

    expected_categories = {"known", "unknown"}
    category_dirs = {
        path.name: path for path in evaluation_dir.iterdir() if path.is_dir()
    }
    for path in category_dirs.values():
        ensure_within(path, evaluation_dir, "Evaluation category directory")
    unexpected_dirs = set(category_dirs) - expected_categories
    missing_dirs = expected_categories - set(category_dirs)
    root_files = [path for path in evaluation_dir.iterdir() if path.is_file()]
    if unexpected_dirs or missing_dirs or root_files:
        raise ValueError(
            "Invalid evaluation directory structure. Expected only known/ and "
            "unknown/ directories, each containing identity directories. "
            f"Missing={sorted(missing_dirs)}, unexpected={sorted(unexpected_dirs)}, "
            f"root_files={[path.name for path in root_files]}"
        )

    samples = []
    excluded = []
    known_identity_keys = set()
    unknown_identity_keys = set()

    for category in ("known", "unknown"):
        category_dir = category_dirs[category]
        identity_dirs = sorted(path for path in category_dir.iterdir() if path.is_dir())
        direct_files = [path for path in category_dir.iterdir() if path.is_file()]
        if direct_files:
            raise ValueError(
                f"Images must be inside identity subdirectories: {category_dir}; "
                f"found {[path.name for path in direct_files]}"
            )
        if not identity_dirs:
            raise ValueError(f"No identity directories found under {category_dir}")

        for identity_dir in identity_dirs:
            ensure_within(identity_dir, evaluation_dir, "Evaluation identity directory")
            identity = identity_dir.name.strip()
            key = identity.casefold()
            if not identity:
                raise ValueError(f"Empty identity directory name: {identity_dir}")

            if category == "known":
                known_identity_keys.add(key)
                if key not in reference_identities:
                    raise ValueError(
                        f"Known evaluation identity is absent from gallery: {identity}"
                    )
            else:
                unknown_identity_keys.add(key)
                if key in reference_identities:
                    raise ValueError(
                        f"Unknown evaluation identity appears in gallery: {identity}"
                    )

            nested_dirs = [path for path in identity_dir.iterdir() if path.is_dir()]
            if nested_dirs:
                raise ValueError(
                    "Invalid evaluation directory structure: identity folders "
                    f"must contain files directly, not nested folders: {identity_dir}"
                )

            all_files = sorted(path for path in identity_dir.iterdir() if path.is_file())
            for path in all_files:
                ensure_within(path, evaluation_dir, "Evaluation sample")
            supported = [
                path for path in all_files
                if path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            excluded.extend({
                "path": path.relative_to(evaluation_dir).as_posix(),
                "category": category,
                "identity": identity,
                "reason": "unsupported_extension",
            } for path in all_files if path not in supported)

            for path in supported:
                samples.append({
                    "path": path,
                    "relative_path": path.relative_to(evaluation_dir).as_posix(),
                    "category": category,
                    "identity": identity,
                })

    overlap = known_identity_keys & unknown_identity_keys
    if overlap:
        raise ValueError(
            "Known and unknown identity folders overlap: " + ", ".join(sorted(overlap))
        )
    if not samples:
        raise ValueError("Evaluation directory contains no supported image files.")

    reference_paths = {path.resolve() for path in references}
    # SHA-256 detects exact byte copies only; transformed copies are not covered.
    reference_hashes = {sha256_file(path): path.name for path in references}
    seen_query_hashes = {}
    for sample in samples:
        query_path = sample["path"].resolve()
        if query_path in reference_paths:
            raise ValueError(f"Gallery/query image path overlap: {query_path}")

        file_hash = sha256_file(query_path)
        if file_hash in reference_hashes:
            raise ValueError(
                "Gallery/query content overlap: "
                f"{sample['relative_path']} is byte-identical to "
                f"{reference_hashes[file_hash]}"
            )
        if file_hash in seen_query_hashes:
            raise ValueError(
                "Duplicate query image content: "
                f"{sample['relative_path']} and {seen_query_hashes[file_hash]}"
            )
        seen_query_hashes[file_hash] = sample["relative_path"]
        sample["sha256"] = file_hash

    reference_manifest = [
        {
            "path": path.relative_to(reference_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in references
    ]
    return (
        reference_dir, evaluation_dir, references, reference_manifest,
        samples, excluded,
    )


def evaluate(reference_dir, evaluation_dir, output_csv, manifest_path, seed=42):
    (
        reference_dir,
        evaluation_dir,
        reference_paths,
        reference_manifest,
        samples,
        excluded,
    ) = validate_dataset(reference_dir, evaluation_dir)

    output_csv_resolved, manifest_path_resolved = validate_output_paths(
        output_csv, manifest_path, (reference_dir, evaluation_dir)
    )

    rng = random.Random(seed)
    rng.shuffle(samples)
    input_file_count = len(samples) + len(excluded)

    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher()

    gallery = build_gallery(reference_dir, detector, embedder)
    if gallery["errors"] or gallery["pending"]:
        raise ValueError(
            "Gallery contains invalid or ambiguous references; resolve these "
            f"before evaluation. errors={gallery['errors']}, pending={gallery['pending']}"
        )
    if not gallery["names"]:
        raise ValueError("Reference gallery is empty.")

    gallery_name_keys = {name.casefold() for name in gallery["names"]}
    expected_name_keys = {path.stem.strip().casefold() for path in reference_paths}
    if gallery_name_keys != expected_name_keys:
        raise ValueError("Gallery identities differ from validated reference files.")

    result_rows = []
    counters = {
        "attempted": len(samples),
        "evaluated": 0,
        "unreadable": 0,
        "zero_faces": 0,
        "multiple_faces": 0,
        "processing_errors": 0,
    }
    for sample in samples:
        image = cv2.imread(str(sample["path"]))
        row = {
            "category": sample["category"],
            "expected": sample["identity"] if sample["category"] == "known" else "unknown",
            "predicted": "None",
            "status": "excluded",
            "best_score": "",
            "second_score": "",
            "margin": "",
            "relative_path": sample["relative_path"],
            "sha256": sample["sha256"],
            "face_count": "",
            "exclusion_reason": "",
        }

        if image is None:
            counters["unreadable"] += 1
            row["exclusion_reason"] = "unreadable_image"
            excluded.append({
                "path": sample["relative_path"],
                "category": sample["category"],
                "identity": sample["identity"],
                "sha256": sample["sha256"],
                "reason": "unreadable_image",
            })
            result_rows.append(row)
            continue

        recognition = recognize_image(image, gallery, detector, embedder, matcher)
        row["face_count"] = recognition["face_count"]
        if recognition["face_count"] == 0:
            counters["zero_faces"] += 1
            row["exclusion_reason"] = "zero_detected_faces"
            excluded.append({
                "path": sample["relative_path"],
                "category": sample["category"],
                "identity": sample["identity"],
                "sha256": sample["sha256"],
                "reason": "zero_detected_faces",
            })
            result_rows.append(row)
            continue
        if recognition["face_count"] > 1:
            counters["multiple_faces"] += 1
            row["exclusion_reason"] = "multiple_detected_faces"
            excluded.append({
                "path": sample["relative_path"],
                "category": sample["category"],
                "identity": sample["identity"],
                "sha256": sample["sha256"],
                "reason": "multiple_detected_faces",
            })
            result_rows.append(row)
            continue

        prediction = recognition["results"][0]
        row.update({
            "predicted": prediction["identity"] or "None",
            "status": prediction["status"],
            "best_score": prediction.get("best_score", ""),
            "second_score": prediction.get("second_score", ""),
            "margin": prediction.get("margin", ""),
        })
        if prediction["status"] == "error":
            counters["processing_errors"] += 1
            row["exclusion_reason"] = prediction.get("error", "recognition_error")
            excluded.append({
                "path": sample["relative_path"],
                "category": sample["category"],
                "identity": sample["identity"],
                "sha256": sample["sha256"],
                "reason": row["exclusion_reason"],
            })
        else:
            counters["evaluated"] += 1
        result_rows.append(row)

    output_csv = output_csv_resolved
    manifest_path = manifest_path_resolved
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = list(result_rows[0].keys()) if result_rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(result_rows)

    known = [row for row in result_rows if row["category"] == "known"]
    unknown = [row for row in result_rows if row["category"] == "unknown"]
    outcome_counts = {
        status: sum(row["status"] == status for row in result_rows)
        for status in ("recognised", "possible", "unknown")
    }
    known_recognised_correct = sum(
        row["status"] == "recognised"
        and row["predicted"].casefold() == row["expected"].casefold()
        for row in known
    )
    known_recognised_wrong = sum(
        row["status"] == "recognised"
        and row["predicted"] != "None"
        and row["predicted"].casefold() != row["expected"].casefold()
        for row in known
    )
    known_possible_correct = sum(
        row["status"] == "possible"
        and row["predicted"].casefold() == row["expected"].casefold()
        for row in known
    )
    known_possible_wrong = sum(
        row["status"] == "possible"
        and row["predicted"] != "None"
        and row["predicted"].casefold() != row["expected"].casefold()
        for row in known
    )
    unknown_false_acceptance = sum(
        row["category"] == "unknown" and row["status"] == "recognised"
        for row in unknown
    )
    unknown_possible = sum(row["status"] == "possible" for row in unknown)
    category_counts = {}
    for category, rows in (("known", known), ("unknown", unknown)):
        category_counts[category] = {
            "attempted_supported_images": len(rows),
            "evaluated_images": sum(
                row["status"] in ("recognised", "possible", "unknown")
                for row in rows
            ),
            "excluded_attempted_images": sum(
                row["status"] in ("excluded", "error") for row in rows
            ),
            "unsupported_files": sum(
                item["category"] == category
                and item["reason"] == "unsupported_extension"
                for item in excluded
            ),
        }
    exclusion_counts = dict(Counter(item["reason"] for item in excluded))

    manifest = {
        "evaluation_type": "FaceFind single-face known/unknown recognition evaluation",
        "seed": seed,
        "split_definition": (
            "Identity membership is supplied by known/<identity>/ and "
            "unknown/<identity>/ directories; the seed fixes query ordering."
        ),
        "reference_directory": str(reference_dir),
        "evaluation_directory": str(evaluation_dir),
        "gallery_identities": sorted(gallery["names"]),
        "gallery_images": reference_manifest,
        "known_identities": sorted({sample["identity"] for sample in samples if sample["category"] == "known"}),
        "unknown_identities": sorted({sample["identity"] for sample in samples if sample["category"] == "unknown"}),
        "model_names": {
            "detector": Path(DETECTOR_MODEL_PATH).name,
            "embedder": Path(EMBEDDER_MODEL_PATH).name,
        },
        "parameters": {
            "detector_score_threshold": 0.5,
            "detector_nms_threshold": 0.3,
            "detector_max_dimension": detector.max_dimension,
            "matcher_recognition_threshold": matcher.recognition_threshold,
            "matcher_possible_threshold": matcher.possible_threshold,
            "matcher_margin_threshold": matcher.margin_threshold,
            "matcher_possible_margin_threshold": matcher.possible_margin_threshold,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        },
        "counts": {
            **counters,
            "input_files": input_file_count,
            "attempted_supported_images": counters["attempted"],
            "excluded_samples": len(excluded),
            "exclusions_by_reason": exclusion_counts,
            "by_category": category_counts,
            "recognised": outcome_counts["recognised"],
            "possible": outcome_counts["possible"],
            "unknown": outcome_counts["unknown"],
            "known_recognised_correct_identity_predictions": known_recognised_correct,
            "known_recognised_wrong_identity_predictions": known_recognised_wrong,
            "known_possible_correct_identity_predictions": known_possible_correct,
            "known_possible_wrong_identity_predictions": known_possible_wrong,
            "unknown_false_acceptances": unknown_false_acceptance,
            "unknown_possible_matches": unknown_possible,
        },
        "samples": [
            {
                "path": sample["relative_path"],
                "category": sample["category"],
                "identity": sample["identity"],
                "sha256": sample["sha256"],
            }
            for sample in samples
        ],
        "excluded_samples": excluded,
        "denominator_policy": (
            "Attempted counts supported image files. Evaluated counts readable, "
            "single-face samples with a non-error recognition result. Exclusions, "
            "including unsupported files, are listed separately and are not in "
            "outcome counts. Counts are image counts, not rates."
        ),
        "metric_notes": {
            "known_recognised_identity_predictions": (
                "Correct/wrong counts include only confirmed recognised outcomes."
            ),
            "known_possible_identity_predictions": (
                "Separate correct/wrong counts represent uncertain possible outcomes."
            ),
            "unknown_false_acceptances": "Counts recognised outcomes in unknown samples.",
        },
        "duplicate_detection_policy": (
            "SHA-256 detects byte-identical images only. Resized, compressed, or "
            "re-encoded duplicates are not detected."
        ),
        "results_csv": str(output_csv),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nFACEFIND RECOGNITION EVALUATION")
    print("--------------------------------")
    print(f"Total attempted images: {counters['attempted']}")
    print(f"Supported and unsupported files found: {input_file_count}")
    print(f"Successfully evaluated images: {counters['evaluated']}")
    print(f"Unreadable images: {counters['unreadable']}")
    print(f"Images with zero detected faces: {counters['zero_faces']}")
    print(f"Images with multiple detected faces: {counters['multiple_faces']}")
    print(f"Recognition processing errors: {counters['processing_errors']}")
    print(f"Recognised outcomes: {outcome_counts['recognised']}")
    print(f"Possible outcomes: {outcome_counts['possible']}")
    print(f"Unknown outcomes: {outcome_counts['unknown']}")
    print(f"Known recognised, correct identity: {known_recognised_correct}")
    print(f"Known recognised, wrong identity: {known_recognised_wrong}")
    print(f"Known possible, correct identity: {known_possible_correct}")
    print(f"Known possible, wrong identity: {known_possible_wrong}")
    print(f"Unknown false acceptances (recognised as known): {unknown_false_acceptance}")
    print(f"Unknown possible matches: {unknown_possible}")
    print(f"Excluded samples listed in manifest: {len(excluded)}")
    print(f"Detailed results: {output_csv}")
    print(f"Evaluation manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate FaceFind recognition without tuning production thresholds."
    )
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=EVALUATION_OUTPUT_DIR / "evaluation_results.csv",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=EVALUATION_OUTPUT_DIR / "evaluation_manifest.json",
        help="Path for the reproducible JSON split and results manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        args.reference_dir,
        args.evaluation_dir,
        args.output,
        args.manifest,
        args.seed,
    )
