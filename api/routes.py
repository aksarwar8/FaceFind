
from pathlib import Path
import base64
import time

from face_engine.pipeline import recognize_image

import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from face_engine.configuration import PIPELINE_LABELS, activate_pipeline


router = APIRouter(prefix="/api", tags=["FaceFind API"])

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

MAX_FILE_SIZE = 15 * 1024 * 1024



def annotate_faces(image, results):
    """Draw recognition results on the group photograph."""

    annotated = image.copy()

    for result in results:
        x, y, width, height = map(
            int, result["bbox"]
        )

        status = result["status"]
        identity = result["identity"]

        if status == "recognised":
            color = (40, 200, 100)
            label = identity

        elif status == "possible":
            color = (0, 165, 255)
            label = f"Possible: {identity}"

        elif status == "unknown":
            color = (60, 60, 230)
            label = "Unknown"

        else:
            color = (128, 128, 128)
            label = "Processing error"

        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            color,
            2
        )

        cv2.putText(
            annotated,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return annotated


class FaceSelection(BaseModel):
    pending_id: int
    face_index: int


class PendingAction(BaseModel):
    pending_id: int


def add_embedding(gallery, identity, embedding):
    """Add one person's face embedding to the gallery."""

    if identity in gallery["names"]:
        raise ValueError(f"Duplicate identity: {identity}")

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != gallery["embeddings"].shape[1] or not np.isfinite(vector).all():
        raise ValueError("Embedding dimensions or values do not match this gallery configuration.")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot cache an empty face embedding.")
    vector = vector / norm

    gallery["names"].append(identity)
    gallery["embeddings"] = np.vstack(
        [gallery["embeddings"], vector.reshape(1, -1)]
    ).astype(np.float32)


def get_gallery_status(request: Request):
    gallery = request.app.state.gallery

    return {
        "gallery_loaded": bool(gallery and gallery["names"]),
        "people_count": len(gallery["names"]) if gallery else 0,
        "pending_references": len(request.app.state.pending_references),
        "pending": (gallery or {}).get("pending", []),
        "gallery_errors": request.app.state.gallery_errors,
        "active_pipeline_id": request.app.state.active_pipeline_id,
        "active_pipeline_label": request.app.state.pipeline_bundles[request.app.state.active_pipeline_id]["label"],
        "gallery_pipeline_id": gallery.get("pipeline_id") if gallery else None,
        "gallery_timings": request.app.state.gallery_timings,
        "gallery_build_errors": request.app.state.gallery_build_errors,
    }


class PipelineSelection(BaseModel):
    pipeline_id: str


@router.get("/health")
def health_check(request: Request):
    return {
        "status": "healthy",
        "application": "FaceFind",
        "engine_initialized": hasattr(request.app.state, "detector"),
    }


@router.get("/status")
def application_status(request: Request):
    return get_gallery_status(request)


@router.get("/pipeline")
def get_pipeline(request: Request):
    return {
        "active_pipeline_id": request.app.state.active_pipeline_id,
        "active_pipeline_label": request.app.state.pipeline_bundles[request.app.state.active_pipeline_id]["label"],
        "available_pipelines": [{"id": key, "label": label} for key, label in PIPELINE_LABELS.items()],
        "model_initialization_ms": request.app.state.pipeline_bundles[request.app.state.active_pipeline_id]["model_initialization_ms"],
        "gallery_loaded": bool(request.app.state.gallery and request.app.state.gallery["names"]),
        "gallery_pipeline_id": request.app.state.gallery.get("pipeline_id") if request.app.state.gallery else None,
    }


@router.post("/pipeline")
def select_pipeline(selection: PipelineSelection, request: Request):
    if selection.pipeline_id not in PIPELINE_LABELS:
        raise HTTPException(400, "Unknown pipeline configuration.")
    try:
        activate_pipeline(request.app, selection.pipeline_id)
    except Exception as error:
        raise HTTPException(503, f"Could not initialize {PIPELINE_LABELS[selection.pipeline_id]}: {error}")
    request.app.state.gallery_timings = (request.app.state.gallery or {}).get("timings", {})
    request.app.state.gallery_errors = (request.app.state.gallery or {}).get("errors", [])
    request.app.state.gallery_build_errors = (request.app.state.gallery or {}).get("errors", [])
    return {"message": "Recognition pipeline selected.", **get_gallery_status(request),
            "model_initialization_ms": request.app.state.pipeline_bundles[selection.pipeline_id]["model_initialization_ms"]}


@router.post("/gallery")
async def upload_gallery(
    request: Request,
    files: list[UploadFile] = File(...)
):
    gallery_started = time.perf_counter()
    detector = request.app.state.detector
    embedder = request.app.state.embedder
    pipeline_id = request.app.state.active_pipeline_id

    gallery = {
        "names": [],
        "embeddings": np.empty((0, int(getattr(embedder, "dimension", 128))), dtype=np.float32),
        "errors": [],
        "pending": [],
        "pipeline_id": pipeline_id,
    }

    pending_references = []
    seen_identities = set()
    embedding_ms = []
    gallery_alignment_ms = []
    gallery_model_embedding_ms = []
    detection_ms = []
    per_reference = []

    if not files:
        raise HTTPException(400, "No reference images uploaded.")

    for uploaded_file in files:
        filename = Path(uploaded_file.filename or "").name
        extension = Path(filename).suffix.lower()
        identity = Path(filename).stem

        if extension not in SUPPORTED_EXTENSIONS:
            gallery["errors"].append(
                f"{filename}: Unsupported file format"
            )
            per_reference.append({"identity": filename, "status": "rejected", "error": "Unsupported file format"})
            continue

        if not identity:
            gallery["errors"].append(
                f"{filename}: Invalid filename"
            )
            per_reference.append({"identity": filename, "status": "rejected", "error": "Invalid filename"})
            continue

        if identity.casefold() in {name.casefold() for name in seen_identities}:
            gallery["errors"].append(
                f"{filename}: Duplicate identity"
            )
            per_reference.append({"identity": identity, "status": "rejected", "error": "Duplicate identity"})
            continue

        seen_identities.add(identity)

        try:
            contents = await uploaded_file.read()

            if not contents:
                gallery["errors"].append(
                    f"{filename}: Empty file"
                )
                per_reference.append({"identity": identity, "status": "rejected", "error": "Empty file"})
                continue

            if len(contents) > MAX_FILE_SIZE:
                gallery["errors"].append(
                    f"{filename}: File exceeds 15 MB"
                )
                per_reference.append({"identity": identity, "status": "rejected", "error": "File exceeds 15 MB"})
                continue

            image = cv2.imdecode(
                np.frombuffer(contents, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if image is None:
                gallery["errors"].append(
                    f"{filename}: Could not read image"
                )
                per_reference.append({"identity": identity, "status": "rejected", "error": "Could not read image"})
                continue

            detect_started = time.perf_counter()
            faces = detector.detect(image)
            detection_ms.append((time.perf_counter() - detect_started) * 1000)

            if len(faces) == 0:
                gallery["errors"].append(
                    f"{filename}: No face detected"
                )
                per_reference.append({"identity": identity, "status": "rejected", "error": "No face detected"})
                continue

            if len(faces) == 1:
                embed_started = time.perf_counter()
                embedding = embedder.get_embedding(
                    image, faces[0]
                )
                elapsed = (time.perf_counter() - embed_started) * 1000
                embedding_ms.append(elapsed)
                component_times = getattr(embedder, "last_timings_ms", {})
                gallery_alignment_ms.append(float(component_times.get("alignment", 0.0)))
                gallery_model_embedding_ms.append(float(component_times.get("embedding", elapsed)))

                add_embedding(
                    gallery, identity, embedding
                )
                per_reference.append({"identity": identity, "status": "accepted", "embedding_ms": elapsed,
                                      "alignment_ms": gallery_alignment_ms[-1],
                                      "model_embedding_ms": gallery_model_embedding_ms[-1]})

            else:
                pending_id = len(pending_references)

                pending_references.append({
                    "pending_id": pending_id,
                    "identity": identity,
                    "filename": filename,
                    "image": image,
                    "faces": faces,
                })

                gallery["pending"].append({
                    "pending_id": pending_id,
                    "identity": identity,
                    "filename": filename,
                    "faces": len(faces),
                })
                per_reference.append({"identity": identity, "status": "manual_selection", "face_count": len(faces)})

        except Exception as error:
            gallery["errors"].append(
                f"{filename}: {str(error)}"
            )
            per_reference.append({"identity": identity or filename, "status": "rejected", "error": str(error)})

        finally:
            await uploaded_file.close()

    gallery["timings"] = {
        "total_gallery_loading_ms": (time.perf_counter() - gallery_started) * 1000,
        "gallery_detection_total_ms": float(sum(detection_ms)),
        "gallery_embedding_total_ms": float(sum(embedding_ms)),
        "gallery_alignment_total_ms": float(sum(gallery_alignment_ms)),
        "gallery_model_embedding_total_ms": float(sum(gallery_model_embedding_ms)),
        "gallery_embedding_per_identity_ms": embedding_ms,
        "accepted_references": len(gallery["names"]),
        "rejected_references": sum(row["status"] == "rejected" for row in per_reference),
        "manual_selection_references": sum(row["status"] == "manual_selection" for row in per_reference),
        "submitted_reference_files": len(files),
        "initial_reference_outcomes": len(per_reference),
        "unaccounted_reference_files": max(0, len(files) - len(per_reference)),
        "per_reference": per_reference,
    }
    previous_gallery = request.app.state.gallery
    if gallery["names"]:
        request.app.state.gallery = gallery
        request.app.state.gallery_cache[pipeline_id] = gallery
        request.app.state.pending_references = pending_references
        request.app.state.pending_cache[pipeline_id] = pending_references
        request.app.state.gallery_errors = gallery["errors"]
        request.app.state.gallery_build_errors = gallery["errors"]
    else:
        # Failed/empty replacement uploads never evict a working gallery.
        if previous_gallery and previous_gallery.get("pipeline_id") == pipeline_id:
            pending_references = request.app.state.pending_references + pending_references
            for index, item in enumerate(pending_references):
                item["pending_id"] = index
            request.app.state.pending_references = pending_references
            request.app.state.pending_cache[pipeline_id] = pending_references
            previous_gallery["pending"] = [
                {"pending_id": item["pending_id"], "identity": item["identity"],
                 "filename": item["filename"], "faces": len(item["faces"])}
                for item in pending_references
            ]
            request.app.state.gallery_errors = gallery["errors"]
            request.app.state.gallery_build_errors = gallery["errors"]
        elif pending_references:
            gallery["pending"] = [
                {"pending_id": item["pending_id"], "identity": item["identity"],
                 "filename": item["filename"], "faces": len(item["faces"])}
                for item in pending_references
            ]
            request.app.state.gallery = gallery
            request.app.state.gallery_cache[pipeline_id] = gallery
            request.app.state.pending_references = pending_references
            request.app.state.pending_cache[pipeline_id] = pending_references
            request.app.state.gallery_errors = gallery["errors"]
            request.app.state.gallery_build_errors = gallery["errors"]
        else:
            request.app.state.gallery_build_errors = gallery["errors"]
    request.app.state.gallery_timings = (request.app.state.gallery or {}).get("timings", gallery["timings"])

    return {
        "message": "Reference gallery processed." if gallery["names"] else "No references were accepted; any existing working gallery was retained.",
        "build_succeeded": bool(gallery["names"]),
        "upload_summary": gallery["timings"],
        **get_gallery_status(request),
        "pending": (request.app.state.gallery or {}).get("pending", []),
    }


@router.post("/gallery/select")
def select_reference_face(
    selection: FaceSelection,
    request: Request
):
    selection_started = time.perf_counter()
    gallery = request.app.state.gallery
    pending = request.app.state.pending_references

    pending_id = selection.pending_id

    if gallery is None:
        raise HTTPException(400, "Upload a reference gallery first.")

    if not 0 <= pending_id < len(pending):
        raise HTTPException(404, "Pending reference not found.")

    reference = pending[pending_id]

    if reference["pending_id"] != pending_id:
        raise HTTPException(404, "Pending reference not found.")

    faces = reference["faces"]

    if not 0 <= selection.face_index < len(faces):
        raise HTTPException(400, "Invalid face selection.")

    identity = reference["identity"]

    if identity in gallery["names"]:
        raise HTTPException(409, "Identity already exists.")

    try:
        embedding_started = time.perf_counter()
        embedding = request.app.state.embedder.get_embedding(
            reference["image"],
            faces[selection.face_index]
        )
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        add_embedding(gallery, identity, embedding)

        component_times = getattr(request.app.state.embedder, "last_timings_ms", {})
        timings = gallery.setdefault("timings", {})
        timings["accepted_references"] = len(gallery["names"])
        timings["manual_selection_embedding_total_ms"] = float(timings.get("manual_selection_embedding_total_ms", 0.0) + embedding_ms)
        timings["gallery_embedding_total_ms"] = float(timings.get("gallery_embedding_total_ms", 0.0) + embedding_ms)
        timings["gallery_alignment_total_ms"] = float(timings.get("gallery_alignment_total_ms", 0.0) + component_times.get("alignment", 0.0))
        timings["gallery_model_embedding_total_ms"] = float(timings.get("gallery_model_embedding_total_ms", 0.0) + component_times.get("embedding", embedding_ms))
        timings.setdefault("gallery_embedding_per_identity_ms", []).append(embedding_ms)
        timings.setdefault("per_reference", []).append({
            "identity": identity, "status": "accepted_after_manual_selection",
            "embedding_ms": embedding_ms,
            "alignment_ms": float(component_times.get("alignment", 0.0)),
            "model_embedding_ms": float(component_times.get("embedding", embedding_ms)),
            "selected_face_index": selection.face_index,
        })

    except Exception as error:
        raise HTTPException(
            500, f"Could not process selected face: {error}"
        )

    pending.pop(pending_id)

    # Keep IDs consistent after removing an item.
    for index, item in enumerate(pending):
        item["pending_id"] = index

    gallery["pending"] = [
        {
            "pending_id": item["pending_id"],
            "identity": item["identity"],
            "filename": item["filename"],
            "faces": len(item["faces"]),
        }
        for item in pending
    ]

    return {
        "message": f"Face added for {identity}.",
        "selection_timing_ms": {"embedding": embedding_ms, "total": (time.perf_counter() - selection_started) * 1000},
        **get_gallery_status(request),
        "pending": gallery["pending"],
    }


@router.post("/gallery/skip")
def skip_reference(
    action: PendingAction,
    request: Request
):
    pending = request.app.state.pending_references

    pending_id = action.pending_id

    if not 0 <= pending_id < len(pending):
        raise HTTPException(404, "Pending reference not found.")

    skipped = pending.pop(pending_id)

    for index, item in enumerate(pending):
        item["pending_id"] = index

    gallery = request.app.state.gallery
    gallery["pending"] = [
        {
            "pending_id": item["pending_id"],
            "identity": item["identity"],
            "filename": item["filename"],
            "faces": len(item["faces"]),
        }
        for item in pending
    ]

    return {
        "message": f"Skipped reference: {skipped['identity']}.",
        **get_gallery_status(request),
        "pending": gallery["pending"],
    }


@router.post("/gallery/clear")
def clear_gallery(request: Request):
    pipeline_id = request.app.state.active_pipeline_id
    request.app.state.gallery = None
    request.app.state.gallery_cache.pop(pipeline_id, None)
    request.app.state.pending_references = []
    request.app.state.pending_cache.pop(pipeline_id, None)
    request.app.state.gallery_errors = []
    request.app.state.gallery_build_errors = []
    request.app.state.gallery_timings = {}

    return {
        "message": "Reference gallery cleared.",
        **get_gallery_status(request),
    }



@router.post("/analyze")
async def analyze_group_photo(
    request: Request,
    file: UploadFile = File(...)
):
    request_started = time.perf_counter()
    # Check whether a reference gallery exists.
    gallery = request.app.state.gallery

    if gallery is None or not gallery["names"]:
        raise HTTPException(
            400,
            "Please upload a reference gallery first."
        )
    if gallery.get("pipeline_id") != request.app.state.active_pipeline_id:
        raise HTTPException(409, "The loaded gallery belongs to a different model configuration. Rebuild it for the selected pipeline.")

    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            "Unsupported image format."
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(400, "Uploaded image is empty.")

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                413,
                "Group photograph exceeds 15 MB."
            )

        image = cv2.imdecode(
            np.frombuffer(contents, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(
                400,
                "Could not decode the uploaded image."
            )

        # Run the shared recognition pipeline.
        recognition = recognize_image(
            image,
            gallery,
            request.app.state.detector,
            request.app.state.embedder,
            request.app.state.matcher
        )

        results = recognition["results"]

        # Draw recognition results.
        annotate_started = time.perf_counter()
        annotated = annotate_faces(image, results)

        # Encode annotated image as JPEG.
        success, encoded_image = cv2.imencode(
            ".jpg", annotated
        )

        if not success:
            raise HTTPException(
                500,
                "Could not encode annotated image."
            )

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")
        annotation_ms = (time.perf_counter() - annotate_started) * 1000

        return {
            "message": "Group photograph analysed successfully.",
            "filename": filename,
            "face_count": recognition["face_count"],
            "summary": {
                "recognised": sum(
                    r["status"] == "recognised" for r in results
                ),
                "possible": sum(
                    r["status"] == "possible" for r in results
                ),
                "unknown": sum(
                    r["status"] == "unknown" for r in results
                ),
                "errors": sum(
                    r["status"] == "error" for r in results
                ),
            },
            "pipeline_id": request.app.state.active_pipeline_id,
            "timings_ms": {
                **recognition["timings_ms"],
                "annotation_and_rendering": annotation_ms,
                "jpeg_encode_and_base64_included": True,
                "backend_total": (time.perf_counter() - request_started) * 1000,
                "gallery_embeddings_reused": True,
                "gallery_embedding_time_during_analysis": 0.0,
            },
            "results": results,
            "annotated_image": (
                f"data:image/jpeg;base64,{image_base64}"
            )
        }

    finally:
        await file.close()



@router.get("/gallery/pending/{pending_id}")
def get_pending_reference(pending_id: int, request: Request):
    pending = request.app.state.pending_references

    if not 0 <= pending_id < len(pending):
        raise HTTPException(404, "Pending reference not found.")

    reference = pending[pending_id]
    image = reference["image"].copy()

    for index, face in enumerate(reference["faces"]):
        x, y, width, height = map(int, face["bbox"])

        cv2.rectangle(
            image, (x, y), (x + width, y + height),
            (0, 200, 255), 2
        )

        cv2.putText(
            image, str(index),
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 200, 255), 2
        )

    success, encoded = cv2.imencode(".jpg", image)

    if not success:
        raise HTTPException(500, "Could not encode preview.")

    preview = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return {
        "pending_id": pending_id,
        "identity": reference["identity"],
        "filename": reference["filename"],
        "faces": len(reference["faces"]),
        "preview": f"data:image/jpeg;base64,{preview}"
    }
