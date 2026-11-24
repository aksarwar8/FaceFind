"""Evaluate the production detector on WIDER FACE single-face images."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_widerface import evaluate as evaluate_widerface
from evaluation.widerface_common import EVALUATION_OUTPUT_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FaceFind detection on WIDER FACE images with an exact "
            "number of valid ground-truth faces."
        )
    )
    parser.add_argument(
        "--exact-faces", type=int, default=1,
        help="Include images with exactly this many valid ground-truth boxes.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Seeded sample size after filtering; use 0 for all eligible images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--manifest", type=Path,
        default=EVALUATION_OUTPUT_DIR / "widerface_single_face_manifest.json",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--max-dimension", type=int, default=1280)
    args = parser.parse_args(argv)

    if args.exact_faces <= 0:
        parser.error("--exact-faces must be a positive integer.")
    if args.limit < 0:
        parser.error("--limit must be zero (all eligible images) or positive.")

    evaluate_widerface(
        limit=args.limit,
        seed=args.seed,
        iou_threshold=args.iou,
        manifest_path=args.manifest,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_dimension=args.max_dimension,
        exact_faces=args.exact_faces,
    )


if __name__ == "__main__":
    main()
