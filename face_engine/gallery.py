
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def build_gallery(reference_dir, detector, embedder, face_selector=None):
    reference_dir = Path(reference_dir)

    if not reference_dir.is_dir():
        raise NotADirectoryError(reference_dir)

    gallery = {}
    errors = []
    pending = []

    image_paths = sorted(
        path for path in reference_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for image_path in image_paths:
        identity = image_path.stem

        if identity in gallery:
            errors.append(f"{image_path.name}: Duplicate identity")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(f"{image_path.name}: Could not read image")
            continue

        faces = detector.detect(image)

        if len(faces) == 0:
            errors.append(f"{image_path.name}: No face detected")
            continue

        if len(faces) == 1:
            selected_face = faces[0]

        elif face_selector is None:
            pending.append({
                "identity": identity,
                "filename": image_path.name,
                "faces": len(faces),
                "reason": "Manual face selection required",
            })
            continue

        else:
            selected_index = face_selector(
                image_path, image, faces
            )

            if (
                not isinstance(selected_index, int)
                or isinstance(selected_index, bool)
                or not 0 <= selected_index < len(faces)
            ):
                errors.append(
                    f"{image_path.name}: Invalid face selection"
                )
                continue

            selected_face = faces[selected_index]

        try:
            embedding = embedder.get_embedding(
                image, selected_face
            )

            gallery[identity] = embedding

        except Exception as error:
            errors.append(
                f"{image_path.name}: Embedding failed: {error}"
            )

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
        "pending": pending,
    }
