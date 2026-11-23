"""WIDER FACE validation annotations and path validation."""

from pathlib import Path

from evaluation.common import validate_output_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "data/benchmarks/widerface/WIDER_val/WIDER_val/images"
ANNOTATION_FILE = (
    PROJECT_ROOT / "data/benchmarks/widerface/annotations/"
    "wider_face_split/wider_face_val_bbx_gt.txt"
)
DATASET_ROOT = PROJECT_ROOT / "data/benchmarks/widerface"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "data/private/evaluation"


def validate_image_path(image_path):
    relative_path = Path(image_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"Annotation image path must stay relative to image root: {image_path}"
        )
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
    return validate_output_path(
        manifest_path,
        EVALUATION_OUTPUT_DIR,
        protected_roots=(DATASET_ROOT, ANNOTATION_FILE),
    )


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
