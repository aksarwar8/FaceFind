
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.match import FaceMatcher
import time


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
    detection_started = time.perf_counter()
    detections = detector.detect(image)
    detection_ms = (time.perf_counter() - detection_started) * 1000

    results = []

    # Step 2: Process each detected face.
    embedding_times = []
    alignment_times = []
    matching_times = []
    for index, detection in enumerate(detections):

        try:
            # Step 3: Extract the face embedding.
            embed_started = time.perf_counter()
            embedding = embedder.get_embedding(image, detection)
            full_embedding_ms = (time.perf_counter() - embed_started) * 1000
            component_times = getattr(embedder, "last_timings_ms", None)
            if component_times:
                alignment_times.append(float(component_times.get("alignment", 0.0)))
                embedding_times.append(float(component_times.get("embedding", full_embedding_ms)))
            else:
                alignment_times.append(0.0)
                embedding_times.append(full_embedding_ms)

            # Step 4: Find the closest identity in the gallery.
            match_started = time.perf_counter()
            prediction = matcher.match(embedding, gallery)
            matching_times.append((time.perf_counter() - match_started) * 1000)

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

    # Resolve repeated gallery identities only after every face has an independent
    # prediction. In a score tie, keep the earliest detected face deterministically.
    identity_groups = {}
    for result in results:
        if (
            result.get("status") in {"recognised", "possible"}
            and result.get("identity") is not None
        ):
            identity_groups.setdefault(result["identity"], []).append(result)

    for conflicting_results in identity_groups.values():
        if len(conflicting_results) < 2:
            continue
        retained = min(
            conflicting_results,
            key=lambda result: (-result["best_score"], result["face_number"]),
        )
        for result in conflicting_results:
            if result is retained:
                continue
            duplicate_status = getattr(matcher, "duplicate_status", None)
            if callable(duplicate_status):
                result["status"] = duplicate_status(result)
                if result["status"] == "unknown":
                    result["identity"] = None
            else:
                result["status"] = (
                    "possible"
                    if result["best_score"] >= matcher.possible_threshold
                    else "unknown"
                )
            result["conflict_with"] = retained["face_number"]

    return {
        "face_count": len(detections),
        "detections": detections,
        "results": results,
        "timings_ms": {
            "detection": detection_ms,
            "face_alignment_total": float(sum(alignment_times)),
            "face_alignment_per_face": alignment_times,
            "face_embedding_total": float(sum(embedding_times)),
            "face_embedding_per_face": embedding_times,
            "identity_matching_total": float(sum(matching_times)),
            "identity_matching_per_face": matching_times,
            "backend_pipeline_total": detection_ms + sum(alignment_times) + sum(embedding_times) + sum(matching_times),
        },
    }
