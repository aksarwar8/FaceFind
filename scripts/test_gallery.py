
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.gallery import build_gallery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir")
    args = parser.parse_args()

    detector = FaceDetector()
    embedder = FaceEmbedder()

    gallery = build_gallery(
        args.reference_dir,
        detector,
        embedder,
    )

    print("\nREFERENCE GALLERY REPORT")
    print("------------------------")
    print("Successful identities:", len(gallery["names"]))
    print("Embedding matrix shape:", gallery["embeddings"].shape)

    for name in gallery["names"]:
        print("Registered:", name)

    if gallery["errors"]:
        print("\nImages requiring attention:")
        for error in gallery["errors"]:
            print("-", error)

    print("\nGallery test completed.")


if __name__ == "__main__":
    main()
