
import argparse
import sys
from pathlib import Path

import cv2

# Add the project root to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_engine.detector import FaceDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the test photograph")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = FaceDetector()
    faces = detector.detect(image)

    print(f"Faces detected: {len(faces)}")

    for index, face in enumerate(faces, start=1):
        x, y, width, height = face["bbox"]
        confidence = face["confidence"]

        print(
            f"Face {index}: "
            f"Box=({x}, {y}, {width}, {height}), "
            f"Confidence={confidence:.3f}"
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"Face {index}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path = image_path.with_name(
        image_path.stem + "_detected.jpg"
    )

    cv2.imwrite(str(output_path), image)
    print(f"Annotated image saved: {output_path}")


if __name__ == "__main__":
    main()
