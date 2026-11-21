
import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.gallery import build_gallery
from face_engine.match import FaceMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir")
    parser.add_argument("query_image")
    args = parser.parse_args()

    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher()

    # Build the reference gallery.
    gallery = build_gallery(
        args.reference_dir,
        detector,
        embedder,
    )

    print("Reference identities:", len(gallery["names"]))

    if not gallery["names"]:
        raise ValueError("No valid reference identities found.")

    # Load the group photograph.
    image = cv2.imread(args.query_image)

    if image is None:
        raise FileNotFoundError(args.query_image)

    # Detect faces in the group photograph.
    faces = detector.detect(image)

    print("Faces detected:", len(faces))
    print("\nMATCHING RESULTS")
    print("----------------")

    for index, face in enumerate(faces, start=1):
        embedding = embedder.get_embedding(image, face)

        result = matcher.match(embedding, gallery)

        print(f"\nFace {index}")
        print("Status:", result["status"])
        print("Identity:", result["identity"])
        print("Best score:", round(result["best_score"], 3))
        print("Second score:", round(result["second_score"], 3))
        print("Margin:", round(result["margin"], 3))

        print("Top candidates:")
        for candidate in result["candidates"]:
            print(
                f"  {candidate['name']}: "
                f"{candidate['score']:.3f}"
            )


if __name__ == "__main__":
    main()
