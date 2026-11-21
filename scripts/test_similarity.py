
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.similarity import cosine_similarity


def main():
    image_path = Path(sys.argv[1])
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    detector = FaceDetector()
    embedder = FaceEmbedder()

    faces = detector.detect(image)

    embeddings = [
        embedder.get_embedding(image, face)
        for face in faces
    ]

    if not embeddings:
        raise ValueError("No faces detected.")

    # Compare every detected face with every other face.
    embeddings = np.vstack(embeddings)
    count = len(embeddings)

    similarity_matrix = np.zeros((count, count))

    for i in range(count):
        for j in range(count):
            similarity_matrix[i, j] = cosine_similarity(
                embeddings[i],
                embeddings[j],
            )

    print("Similarity matrix shape:", similarity_matrix.shape)
    print("\nSimilarity matrix:")
    print(np.round(similarity_matrix, 3))

    print("\nSelf-similarity:")
    print(
        cosine_similarity(embeddings[0], embeddings[0])
    )

    print("\nSimilarity test PASSED!")


if __name__ == "__main__":
    main()
