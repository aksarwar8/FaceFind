"""Small utilities shared by independent evaluation experiments."""

import json
import os
from pathlib import Path


def calculate_detection_metrics(true_positives, false_positives, false_negatives):
    """Return aggregate precision, recall, and F1 using explicit counts."""
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def validate_output_path(output_path, output_root, protected_roots=()):
    """Validate a new output path is safe and contained in its output root."""
    output = Path(output_path).resolve()
    root_path = Path(output_root)
    resolved_root = root_path.resolve()
    expected_root = Path(os.path.abspath(root_path))

    if resolved_root != expected_root:
        raise ValueError(
            f"Evaluation output directory must not resolve through a symlink: {root_path}"
        )
    if root_path.exists() and not root_path.is_dir():
        raise ValueError(f"Evaluation output path is not a directory: {root_path}")
    try:
        output.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Evaluation output must be inside the dedicated directory {root_path}: {output}"
        ) from error
    if output == resolved_root:
        raise ValueError(f"Output file path cannot be the evaluation directory: {output}")
    if output.exists() and output.is_dir():
        raise ValueError(f"Output path is a directory: {output}")
    if output.exists():
        if output.stat().st_nlink > 1:
            raise ValueError(f"Output path has multiple hard links and is unsafe: {output}")
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {output}")

    for protected_root in protected_roots:
        protected = Path(protected_root).resolve()
        if output == protected or protected in output.parents:
            raise ValueError(f"Output path must be outside protected input data: {output}")
    return output


def write_manifest_exclusive(manifest_path, manifest):
    """Write a JSON manifest without replacing a file created by another run."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
