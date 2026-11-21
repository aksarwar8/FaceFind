
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_gallery(reference_dir, detector, embedder):
    reference_dir = Path(reference_dir)

    if not reference_dir.is_dir():
        raise NotADirectoryError(reference_dir)

    gallery = {}
    errors = []

    image_paths = sorted(
        path for path in reference_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(f"{image_path.name}: Could not read image")
            continue

        faces = detector.detect(image)

        if len(faces) != 1:
            errors.append(
                f"{image_path.name}: Expected 1 face, found {len(faces)}"
            )
            continue

        embedding = embedder.get_embedding(image, faces[0])

        identity = image_path.stem

        if identity in gallery:
            errors.append(f"{image_path.name}: Duplicate identity")
            continue

        gallery[identity] = embedding

    if gallery:
        names = list(gallery.keys())
        embeddings = np.vstack(list(gallery.values()))
    else:
        names = []
        embeddings = np.empty((0, 128), dtype=np.float32)

    return {
        "names": names,
        "embeddings": embeddings,
        "errors": errors,
    }
