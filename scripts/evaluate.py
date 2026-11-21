
from pathlib import Path
import sys
import argparse
import csv

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.gallery import build_gallery
from face_engine.match import FaceMatcher


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate(reference_dir, evaluation_dir, output_csv):
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher()

    gallery = build_gallery(reference_dir, detector, embedder)

    if not gallery["names"]:
        raise ValueError("Reference gallery is empty.")

    print(f"Reference identities: {len(gallery['names'])}")

    results = []

    for category in ["known", "unknown"]:
        category_dir = evaluation_dir / category

        if not category_dir.is_dir():
            print(f"Missing evaluation folder: {category_dir}")
            continue

        for image_path in sorted(category_dir.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            expected = (
                image_path.parent.name if category == "known" else "unknown"
            )

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not read: {image_path.name}")
                continue

            faces = detector.detect(image)

            if len(faces) != 1:
                print(
                    f"Skipped {image_path.name}: "
                    f"expected 1 face, found {len(faces)}"
                )
                continue

            embedding = embedder.get_embedding(image, faces[0])
            prediction = matcher.match(embedding, gallery)

            predicted = prediction["identity"]
            status = prediction["status"]

            results.append({
                "category": category,
                "expected": expected,
                "predicted": predicted or "None",
                "status": status,
                "best_score": round(prediction["best_score"], 4),
                "second_score": round(prediction["second_score"], 4),
                "margin": round(prediction["margin"], 4),
                "filename": image_path.name,
            })

            print(
                f"{image_path.name:30} "
                f"Expected: {expected:22} "
                f"Predicted: {str(predicted):22} "
                f"Status: {status:10} "
                f"Score: {prediction['best_score']:.3f}"
            )

    if not results:
        print("\nNo valid evaluation images found.")
        return

    fields = list(results[0].keys())

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    known = [r for r in results if r["category"] == "known"]
    unknown = [r for r in results if r["category"] == "unknown"]

    correct = sum(
        r["status"] == "recognised"
        and r["predicted"] == r["expected"]
        for r in known
    )

    wrong = sum(
        r["status"] == "recognised"
        and r["predicted"] != r["expected"]
        for r in known
    )

    known_possible = sum(r["status"] == "possible" for r in known)
    known_rejected = sum(r["status"] == "unknown" for r in known)

    unknown_confirmed = sum(
        r["status"] == "recognised" for r in unknown
    )

    unknown_possible = sum(
        r["status"] == "possible" for r in unknown
    )

    unknown_rejected = sum(
        r["status"] == "unknown" for r in unknown
    )

    print("\nEVALUATION SUMMARY")
    print("------------------")
    print(f"Known images: {len(known)}")
    print(f"Correctly recognised: {correct}")
    print(f"Wrong confirmed identity: {wrong}")
    print(f"Possible matches: {known_possible}")
    print(f"Rejected known faces: {known_rejected}")

    print(f"\nUnknown images: {len(unknown)}")
    print(f"Incorrectly confirmed as known: {unknown_confirmed}")
    print(f"Possible matches: {unknown_possible}")
    print(f"Correctly rejected: {unknown_rejected}")

    if known:
        print(f"\nKnown recognition rate: {correct / len(known):.2%}")

    if unknown:
        print(
            "Unknown confirmed false-acceptance rate: "
            f"{unknown_confirmed / len(unknown):.2%}"
        )
        print(
            "Unknown rejection rate: "
            f"{unknown_rejected / len(unknown):.2%}"
        )

    print(f"\nDetailed results saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "reference_dir",
        type=Path,
    )

    parser.add_argument(
        "evaluation_dir",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/private/evaluation_results.csv"),
    )

    args = parser.parse_args()

    evaluate(args.reference_dir, args.evaluation_dir, args.output)
