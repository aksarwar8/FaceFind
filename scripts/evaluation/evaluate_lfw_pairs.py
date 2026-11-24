"""Evaluate FaceFind's YuNet + SFace pipeline on the official LFW pairs protocol.

This standalone experiment records failed image/detection cases explicitly and
never changes production thresholds. The protocol and image tree are expected
under data/benchmarks/lfw unless paths are supplied on the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
from face_engine.similarity import cosine_similarity

DATA_ROOT = PROJECT_ROOT / "data" / "benchmarks" / "lfw"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "private" / "evaluation"
THRESHOLDS = {"opencv_sface_0.363": 0.363, "application_possible_0.35": 0.35,
              "application_recognition_0.50": 0.50}


def load_pairs_protocol(path: Path):
    """Parse the official 10-fold pairs.txt layout into labelled image pairs."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("LFW pairs protocol is empty.")
    try:
        folds, pairs_per_fold = map(int, lines[0].split())
    except (ValueError, TypeError) as error:
        raise ValueError("Unsupported LFW pairs header; expected '<folds> <pairs_per_fold>'.") from error
    expected = 1 + folds * pairs_per_fold * 2
    if len(lines) != expected:
        raise ValueError(f"LFW protocol has {len(lines)-1} rows; expected {expected-1} for its header.")
    pairs = []
    cursor = 1
    for fold in range(1, folds + 1):
        for label in (1, 0):
            for _ in range(pairs_per_fold):
                fields = lines[cursor].split()
                cursor += 1
                if label == 1 and len(fields) == 3:
                    identity, first, second = fields
                    names = (identity, identity)
                elif label == 0 and len(fields) == 4:
                    first_name, first, second_name, second = fields
                    names = (first_name, second_name)
                else:
                    raise ValueError(f"Malformed LFW pair row {cursor}: {lines[cursor-1]}")
                try:
                    image_paths = [
                        f"{name}/{name}_{int(index):04d}.jpg"
                        for name, index in zip(names, (first, second))
                    ]
                except ValueError as error:
                    raise ValueError(f"Invalid image index at protocol row {cursor}.") from error
                pairs.append({"fold": fold, "label": "genuine" if label else "impostor",
                              "same_identity": bool(label), "images": image_paths})
    return {"folds": folds, "pairs_per_fold_per_class": pairs_per_fold,
            "pairs": pairs}


def score_protocol(protocol, image_root, detector, embedder):
    """Extract each referenced image once, then score every official pair."""
    cache = {}
    image_root = Path(image_root).resolve()
    for relative in sorted({p for pair in protocol["pairs"] for p in pair["images"]}):
        image_path = (image_root / relative).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError:
            cache[relative] = {"status": "failed", "reason": "path_outside_image_root"}
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            cache[relative] = {"status": "failed", "reason": "unreadable_image"}
            continue
        try:
            faces = detector.detect(image)
        except Exception as error:
            cache[relative] = {"status": "failed", "reason": "detection_error", "detail": str(error)}
            continue
        if not faces:
            cache[relative] = {"status": "failed", "reason": "no_face_detected"}
            continue
        if len(faces) != 1:
            cache[relative] = {"status": "failed", "reason": "multiple_faces_detected",
                               "face_count": len(faces)}
            continue
        try:
            cache[relative] = {"status": "ok", "embedding": embedder.get_embedding(image, faces[0])}
        except Exception as error:
            cache[relative] = {"status": "failed", "reason": "embedding_error", "detail": str(error)}

    scored = []
    for index, pair in enumerate(protocol["pairs"], 1):
        left, right = (cache[path] for path in pair["images"])
        failure = left if left["status"] != "ok" else right if right["status"] != "ok" else None
        row = {"pair_index": index, "fold": pair["fold"], "label": pair["label"],
               "same_identity": pair["same_identity"], "images": pair["images"]}
        if failure:
            row.update(status="excluded", exclusion_reason=failure["reason"])
        else:
            row.update(status="scored", similarity=cosine_similarity(left["embedding"], right["embedding"]))
        scored.append(row)
    return cache, scored


def confusion(rows, threshold):
    tp = fp = tn = fn = 0
    for row in rows:
        if row["status"] != "scored":
            continue
        predicted_same = row["similarity"] >= threshold
        if row["same_identity"] and predicted_same: tp += 1
        elif not row["same_identity"] and predicted_same: fp += 1
        elif not row["same_identity"]: tn += 1
        else: fn += 1
    total = tp + fp + tn + fn
    return {"threshold": threshold, "accuracy": (tp + tn) / total if total else 0.0,
            "true_positive_rate": tp / (tp + fn) if tp + fn else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "true_positive": tp, "false_positive": fp, "true_negative": tn,
            "false_negative": fn, "evaluated_pairs": total}


def write_histogram(rows, path):
    """Write a dependency-free SVG score histogram, split by pair class."""
    bins = 40
    counts = {"genuine": [0] * bins, "impostor": [0] * bins}
    for row in rows:
        if row["status"] == "scored":
            index = min(bins - 1, max(0, int((row["similarity"] + 1) * bins / 2)))
            counts[row["label"]][index] += 1
    max_count = max((max(values) for values in counts.values()), default=1) or 1
    bars = []
    for i in range(bins):
        x = 55 + i * 20
        for key, color, offset in (("genuine", "#197a3a", 0), ("impostor", "#c43b36", 9)):
            height = counts[key][i] * 280 / max_count
            bars.append(f'<rect x="{x+offset}" y="{320-height:.1f}" width="8" height="{height:.1f}" fill="{color}"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="920" height="390">'
           '<rect width="100%" height="100%" fill="white"/><text x="55" y="25" font-size="18">LFW genuine/impostor similarity scores</text>'
           '<path d="M55 320 H855 M55 40 V320" stroke="black"/>' + ''.join(bars) +
           '<text x="55" y="350">-1.0</text><text x="440" y="350">0.0</text><text x="830" y="350">1.0</text>'
           '<text x="650" y="45" fill="#197a3a">Genuine</text><text x="730" y="45" fill="#c43b36">Impostor</text></svg>')
    Path(path).write_text(svg, encoding="utf-8")


def evaluate(pairs_path, image_root, output_dir):
    protocol = load_pairs_protocol(pairs_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    detector, embedder = FaceDetector(), FaceEmbedder()
    image_records, rows = score_protocol(protocol, image_root, detector, embedder)
    results = {name: confusion(rows, threshold) for name, threshold in THRESHOLDS.items()}
    fold_results = {str(fold): {name: confusion([r for r in rows if r["fold"] == fold], t)
                                for name, t in THRESHOLDS.items()}
                    for fold in range(1, protocol["folds"] + 1)}
    exclusion_counts = Counter(record.get("reason", "") for record in image_records.values()
                               if record["status"] != "ok")
    possible_bins = Counter()
    for row in rows:
        if row["status"] == "scored":
            score = row["similarity"]
            possible_bins["below_0.35" if score < 0.35 else "0.35_to_below_0.50" if score < 0.50 else "at_or_above_0.50"] += 1
    report = {
        "evaluation": "LFW official pairs protocol; FaceFind YuNet + SFace",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(Path(pairs_path).resolve()), "image_root": str(Path(image_root).resolve()),
        "protocol": {"folds": protocol["folds"], "pairs_per_fold_per_class": protocol["pairs_per_fold_per_class"],
                     "total_pairs": len(rows)},
        "detector_parameters": {"score_threshold": 0.5, "nms_threshold": 0.3, "max_dimension": 1280},
        "threshold_results": results, "fold_results": fold_results,
        "application_score_bands": dict(possible_bins),
        "counts": {"scored_pairs": sum(r["status"] == "scored" for r in rows),
                   "excluded_pairs": sum(r["status"] == "excluded" for r in rows),
                   "unique_images": len(image_records), "failed_images_by_reason": dict(exclusion_counts)},
        "image_failures": {path: {k: v for k, v in item.items() if k != "embedding"}
                           for path, item in image_records.items() if item["status"] != "ok"},
        "pair_results": rows,
        "reproducibility": {"code_revision": None, "random_seed": None,
                            "note": "Official folds are fixed by pairs.txt; detector uses FaceDetector defaults."},
    }
    json_path = output_dir / "results.json"
    write_manifest_exclusive(json_path, report)
    write_histogram(rows, output_dir / "score_histogram.svg")
    with (output_dir / "accuracy_comparison.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["threshold_name", *next(iter(results.values())).keys()])
        writer.writeheader()
        for name, metrics in results.items(): writer.writerow({"threshold_name": name, **metrics})
    print(json.dumps({"output_dir": str(output_dir), "counts": report["counts"], "threshold_results": results}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DATA_ROOT / "pairs.txt")
    parser.add_argument("--images-dir", type=Path, default=DATA_ROOT / "lfw_funneled")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New output directory. Defaults to a unique directory under data/private/evaluation.")
    args = parser.parse_args(argv)
    if not args.pairs.is_file():
        parser.error(f"Official LFW pairs protocol not found: {args.pairs}. Obtain it from the official LFW source.")
    if not args.images_dir.is_dir(): parser.error(f"LFW image directory not found: {args.images_dir}")
    if args.output_dir is None:
        args.output_dir = OUTPUT_ROOT / f"lfw_pairs_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    resolved = validate_output_path(
        args.output_dir / "results.json", OUTPUT_ROOT,
        protected_roots=(DATA_ROOT, args.images_dir, args.pairs.parent),
    )
    evaluate(args.pairs, args.images_dir, resolved.parent)


if __name__ == "__main__":
    main()
