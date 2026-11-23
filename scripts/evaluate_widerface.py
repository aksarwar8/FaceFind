"""Backward-compatible CLI for the mixed-density WIDER FACE evaluation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_widerface import (  # noqa: E402,F401
    ANNOTATION_FILE,
    DATASET_ROOT,
    EVALUATION_OUTPUT_DIR,
    IMAGE_ROOT,
    calculate_iou,
    evaluate,
    main,
    validate_image_path,
    validate_manifest_path,
    load_annotations,
)


if __name__ == "__main__":
    main()
