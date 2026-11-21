
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide an image path.")

    image_path = Path(sys.argv[1])
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    detector = FaceDetector()
    embedder = FaceEmbedder()

    faces = detector.detect(image)

    embeddings = []

    for index, face in enumerate(faces, start=1):
        embedding = embedder.get_embedding(image, face)
        embeddings.append(embedding)

        print(
            f"Face {index}: "
            f"Embedding shape={embedding.shape}, "
            f"L2 norm={np.linalg.norm(embedding):.4f}"
        )

    if embeddings:
        embeddings = np.vstack(embeddings)

        print("\nEmbedding matrix shape:", embeddings.shape)
        print("Total embeddings:", len(embeddings))

    print("\nSFace embedding test PASSED!")


if __name__ == "__main__":
    main()
