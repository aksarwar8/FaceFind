
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.match import FaceMatcher


def recognize_image(image, gallery, detector, embedder, matcher):
    """
    Detect faces in an image, extract their embeddings,
    and identify them using the reference gallery.

    Parameters:
        image: OpenCV image (BGR NumPy array)
        gallery: Reference gallery containing names and embeddings
        detector: Initialized FaceDetector
        embedder: Initialized FaceEmbedder
        matcher: Initialized FaceMatcher

    Returns:
        A dictionary containing face detections and recognition results.
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid input image.")

    if not gallery or not gallery["names"]:
        raise ValueError("Reference gallery is empty.")

    # Step 1: Detect all faces in the image.
    detections = detector.detect(image)

    results = []

    # Step 2: Process each detected face.
    for index, detection in enumerate(detections):

        try:
            # Step 3: Extract the face embedding.
            embedding = embedder.get_embedding(image, detection)

            # Step 4: Find the closest identity in the gallery.
            prediction = matcher.match(embedding, gallery)

            # Step 5: Store the prediction and detection details.
            results.append({
                "face_number": index + 1,
                "identity": prediction["identity"],
                "status": prediction["status"],
                "best_score": prediction["best_score"],
                "second_score": prediction["second_score"],
                "margin": prediction["margin"],
                "candidates": prediction["candidates"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
            })

        except Exception as error:
            results.append({
                "face_number": index + 1,
                "identity": None,
                "status": "error",
                "error": str(error),
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
            })

    return {
        "face_count": len(detections),
        "detections": detections,
        "results": results,
    }
