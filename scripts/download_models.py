
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_DIR = Path("models/face")

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "min_size": 100_000,
    },
    "face_recognition_sface_2021dec.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "min_size": 10_000_000,
    },
}


def download_model(filename, details):
    destination = MODEL_DIR / filename
    temporary = destination.with_suffix(".download")

    if destination.exists() and destination.stat().st_size >= details["min_size"]:
        print(f"Already available: {filename}")
        return

    print(f"Downloading {filename}...")

    request = Request(
        details["url"],
        headers={"User-Agent": "FaceFind-Model-Setup"}
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = response.read()

        if len(data) < details["min_size"]:
            raise ValueError(
                f"Downloaded file is too small ({len(data)} bytes). "
                "It may be a Git LFS pointer or an error page."
            )

        temporary.write_bytes(data)
        temporary.replace(destination)

        print(f"Saved: {destination} ({destination.stat().st_size:,} bytes)")

    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename, details in MODELS.items():
        download_model(filename, details)

    print("\nAll model files downloaded and size-checked.")


if __name__ == "__main__":
    main()
