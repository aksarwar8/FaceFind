
from pathlib import Path
import base64
import hashlib
import logging
import time
from typing import Annotated
from uuid import uuid4

from face_engine.pipeline import recognize_image

import cv2
import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from face_engine.configuration import PIPELINE_LABELS, activate_pipeline, model_error_detail
from face_engine.reference_selection import (SMART, ReferenceSelectionPolicy, dominant_face_selection,
                                             valid_reference_face)
from .annotations import annotate_faces
from .images import IMAGE_EXTENSIONS, MAX_PENDING_BYTES, ImageInputError, decode_image, upload_policy
from .upload_route import ImageUploadRoute


router = APIRouter(prefix="/api", tags=["FaceFind API"], route_class=ImageUploadRoute)
logger = logging.getLogger(__name__)


class FaceSelection(BaseModel):
    pending_id: str
    face_index: int
    expected_identity: str | None = None


class PendingAction(BaseModel):
    pending_id: str
    expected_identity: str | None = None


class AutomaticFaceCorrection(BaseModel):
    revision: str
    face_index: int


def normalized_embedding(gallery, embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != gallery["embeddings"].shape[1] or not np.isfinite(vector).all():
        raise ValueError("Embedding dimensions or values do not match this gallery configuration.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize the face embedding.")
    return vector / norm


def add_embedding(gallery, identity, embedding):
    """Add one person's face embedding to the gallery."""
    if any(name.casefold() == identity.casefold() for name in gallery["names"]):
        raise ValueError(f"Duplicate identity: {identity}")

    vector = normalized_embedding(gallery, embedding)
    gallery["embeddings"] = np.vstack(
        [gallery["embeddings"], vector.reshape(1, -1)]
    )
    gallery["names"].append(identity)


def save_thumbnail(gallery, identity, image, face):
    """Keep only a small face crop in the current in-memory gallery."""
    if "bbox" not in face:
        return
    x, y, width, height = map(int, face["bbox"])
    crop = image[max(0, y):max(0, y + height), max(0, x):max(0, x + width)]
    if not crop.size:
        return
    try:
        scale = min(1, 96 / max(crop.shape[:2]))
        crop = cv2.resize(crop, tuple(max(1, int(side * scale)) for side in crop.shape[1::-1]),
                          interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", crop)
        if success:
            gallery.setdefault("thumbnails", {})[identity] = encoded.tobytes()
    except cv2.error:
        return


def get_gallery_status(request: Request):
    gallery = request.app.state.gallery
    active_pipeline_id = request.app.state.active_pipeline_id
    active_bundle = request.app.state.pipeline_bundles.get(active_pipeline_id)

    return {
        "engine_initialized": active_bundle is not None,
        "startup_error": request.app.state.startup_error,
        "image_upload": upload_policy(),
        "fallback_reason": request.app.state.fallback_reason,
        "gallery_loaded": bool(gallery and gallery["names"]),
        "people_count": len(gallery["names"]) if gallery else 0,
        "pending_references": len(request.app.state.pending_references),
        "pending": (gallery or {}).get("pending", []),
        "automatic_references": [
            {key: value for key, value in reference.items() if key not in ("image", "faces", "embedding_index")}
            | {"faces": len(reference["faces"])}
            for reference in (gallery or {}).get("automatic_references", {}).values()
        ],
        "gallery_errors": request.app.state.gallery_errors,
        "active_pipeline_id": request.app.state.active_pipeline_id,
        "active_pipeline_label": active_bundle["label"] if active_bundle else None,
        "gallery_pipeline_id": gallery.get("pipeline_id") if gallery else None,
        "gallery_timings": request.app.state.gallery_timings,
        "gallery_build_errors": request.app.state.gallery_build_errors,
        "gallery_revision": (gallery or {}).get("revision"),
        "identities": (gallery or {}).get("names", []),
        "thumbnail_indices": [
            index for index, name in enumerate((gallery or {}).get("names", []))
            if name in gallery.get("thumbnails", {})
        ],
    }


class PipelineSelection(BaseModel):
    pipeline_id: str


@router.get("/health")
def health_check(request: Request):
    initialized = bool(
        request.app.state.active_pipeline_id
        and request.app.state.active_pipeline_id in request.app.state.pipeline_bundles
    )
    return {
        "status": "healthy" if initialized else "unavailable",
        "application": "FaceFind",
        "engine_initialized": initialized,
        "active_pipeline_id": request.app.state.active_pipeline_id,
        "fallback_reason": request.app.state.fallback_reason,
        "startup_error": request.app.state.startup_error,
    }


@router.get("/status")
def application_status(request: Request):
    return get_gallery_status(request)


@router.get("/gallery/thumbnail/{revision}/{index}")
def get_gallery_thumbnail(revision: str, index: int, request: Request):
    gallery = request.app.state.gallery or {}
    names = gallery.get("names", [])
    if gallery.get("revision") != revision or not 0 <= index < len(names):
        raise HTTPException(404, "Reference thumbnail not found.")
    thumbnail = gallery.get("thumbnails", {}).get(names[index])
    if thumbnail is None:
        raise HTTPException(404, "Reference thumbnail not found.")
    return Response(thumbnail, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/pipeline")
def get_pipeline(request: Request):
    active_id = request.app.state.active_pipeline_id
    active_bundle = request.app.state.pipeline_bundles.get(active_id)
    models = {
        "scrfd_adaface": ("SCRFD-10G-KPS", "AdaFace IR-18"),
        "yunet_sface": ("YuNet", "SFace"),
    }
    detector, recognizer = models.get(active_id, (None, None))
    return {
        "requested_pipeline_id": getattr(request.app.state, "requested_pipeline_id", None),
        "detector": detector,
        "recognizer": recognizer,
        "embedding_dimensions": getattr(active_bundle["embedder"], "dimension", None) if active_bundle else None,
        "device": "CPU" if active_bundle else None,
        "initialized_pipelines": list(request.app.state.pipeline_bundles),
        "active_pipeline_id": active_id,
        "active_pipeline_label": active_bundle["label"] if active_bundle else None,
        "available_pipelines": [{"id": key, "label": label} for key, label in PIPELINE_LABELS.items()],
        "model_initialization_ms": active_bundle["model_initialization_ms"] if active_bundle else None,
        "engine_initialized": active_bundle is not None,
        "fallback_reason": request.app.state.fallback_reason,
        "startup_error": request.app.state.startup_error,
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
        raise HTTPException(503, f"Could not initialize {PIPELINE_LABELS[selection.pipeline_id]}: {model_error_detail(error)}")
    request.app.state.startup_error = None
    request.app.state.requested_pipeline_id = selection.pipeline_id
    request.app.state.fallback_reason = None
    request.app.state.gallery_timings = (request.app.state.gallery or {}).get("timings", {})
    request.app.state.gallery_errors = (request.app.state.gallery or {}).get("errors", [])
    request.app.state.gallery_build_errors = (request.app.state.gallery or {}).get("errors", [])
    return {"message": "Recognition pipeline selected.", **get_gallery_status(request),
            "model_initialization_ms": request.app.state.pipeline_bundles[selection.pipeline_id]["model_initialization_ms"]}


@router.post("/gallery")
async def upload_gallery(
    request: Request,
    files: list[UploadFile] = File(...),
    reference_selection_policy: Annotated[ReferenceSelectionPolicy, Form()] = SMART,
):
    gallery_started = time.perf_counter()
    detector = request.app.state.detector
    embedder = request.app.state.embedder
    pipeline_id = request.app.state.active_pipeline_id
    if not pipeline_id:
        raise HTTPException(503, "The recognition engine is unavailable.")

    gallery = {
        "names": [],
        "embeddings": np.empty((0, int(getattr(embedder, "dimension", 128))), dtype=np.float32),
        "errors": [],
        "pending": [],
        "automatic_references": {},
        "pipeline_id": pipeline_id,
        "revision": uuid4().hex,
    }

    pending_references = []
    review_bytes = sum(
        reference["image"].nbytes
        for queue in request.app.state.pending_cache.values()
        for reference in queue
    )
    review_bytes += sum(
        reference["image"].nbytes
        for cached in request.app.state.gallery_cache.values()
        for reference in cached.get("automatic_references", {}).values()
    )
    seen_identities = set()
    embedding_ms = []
    gallery_alignment_ms = []
    gallery_model_embedding_ms = []
    detection_ms = []
    reference_detection = getattr(detector, "detect_reference", None)
    retry_count = 0
    retry_recovered_count = 0
    retry_total_ms = 0.0
    per_reference = []

    if not files:
        raise HTTPException(400, "No reference images uploaded.")

    for uploaded_file in files:
        filename = Path((uploaded_file.filename or "").replace("\\", "/")).name
        identity = Path(filename).stem

        if not identity.strip() or filename.lower() in IMAGE_EXTENSIONS or identity in (".", ".."):
            gallery["errors"].append(
                f"{filename}: Invalid filename"
            )
            per_reference.append({"identity": filename, "status": "rejected", "error": "Invalid filename"})
            await uploaded_file.close()
            continue

        if identity.casefold() in seen_identities:
            gallery["errors"].append(
                f"{filename}: Duplicate identity"
            )
            per_reference.append({"identity": identity, "status": "rejected", "error": "Duplicate identity"})
            await uploaded_file.close()
            continue

        seen_identities.add(identity.casefold())

        detection = {}
        try:
            # Existing OpenCV and AdaFace adapters consume BGR.
            image = cv2.cvtColor(decode_image(uploaded_file), cv2.COLOR_RGB2BGR)

            detect_started = time.perf_counter()
            if reference_detection is not None:
                faces, diagnostics = reference_detection(image)
                detection["detection"] = diagnostics
                if diagnostics["retry_ms"] is not None:
                    retry_count += 1
                    retry_recovered_count += bool(faces)
                    retry_total_ms += diagnostics["retry_ms"]
            else:
                faces = detector.detect(image)
            detection_ms.append((time.perf_counter() - detect_started) * 1000)
            faces = [face for face in faces if valid_reference_face(face)]

            if len(faces) == 0:
                reason = ("No face detected at either SCRFD scale (1280 and 640 pixels)"
                          if detection else "No face detected")
                gallery["errors"].append(
                    f"{filename}: {reason}"
                )
                per_reference.append({"identity": identity, "status": "rejected", "error": reason, **detection})
                continue

            automatic = dominant_face_selection(faces, reference_selection_policy)
            if len(faces) > 1 and review_bytes + image.nbytes > MAX_PENDING_BYTES:
                raise ImageInputError("Reference review storage is full. Clear an unused gallery or resolve pending references and retry.")

            if len(faces) == 1 or automatic is not None:
                face = faces[automatic["selected_face_index"]] if automatic else faces[0]
                embed_started = time.perf_counter()
                embedding = embedder.get_embedding(
                    image, face
                )
                elapsed = (time.perf_counter() - embed_started) * 1000
                embedding_ms.append(elapsed)
                component_times = getattr(embedder, "last_timings_ms", {})
                gallery_alignment_ms.append(float(component_times.get("alignment", 0.0)))
                gallery_model_embedding_ms.append(float(component_times.get("embedding", elapsed)))

                add_embedding(
                    gallery, identity, embedding
                )
                save_thumbnail(gallery, identity, image, face)
                if automatic:
                    review_bytes += image.nbytes
                    review_id = uuid4().hex
                    gallery["automatic_references"][review_id] = {
                        "pending_id": review_id, "embedding_index": len(gallery["names"]) - 1, "identity": identity, "filename": filename,
                        "image": image, "faces": faces, "automatic_selection": automatic,
                        "selected_face_index": automatic["selected_face_index"],
                    }
                per_reference.append({"identity": identity, "status": "accepted", "embedding_ms": elapsed,
                                      "alignment_ms": gallery_alignment_ms[-1],
                                      "model_embedding_ms": gallery_model_embedding_ms[-1],
                                      "face_count": len(faces),
                                      **({"automatic_selection": automatic} if automatic else {}), **detection})

            else:
                review_bytes += image.nbytes
                pending_id = uuid4().hex

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
                per_reference.append({"identity": identity, "status": "manual_selection", "face_count": len(faces), **detection})

        except ImageInputError as error:
            gallery["errors"].append(
                f"{filename}: {str(error)}"
            )
            per_reference.append({"identity": identity or filename, "status": "rejected", "error": str(error), **detection})
        except Exception:
            logger.exception("Could not process a reference image.")
            reason = "Could not process the reference image."
            gallery["errors"].append(f"{filename}: {reason}")
            per_reference.append({"identity": identity or filename, "status": "rejected", "error": reason, **detection})

        finally:
            await uploaded_file.close()

    gallery["timings"] = {
        "total_gallery_loading_ms": (time.perf_counter() - gallery_started) * 1000,
        "gallery_detection_total_ms": float(sum(detection_ms)),
        "gallery_detection_retry_count": retry_count,
        "gallery_detection_retry_recovered_count": retry_recovered_count,
        "gallery_detection_retry_total_ms": retry_total_ms,
        "gallery_embedding_total_ms": float(sum(embedding_ms)),
        "gallery_alignment_total_ms": float(sum(gallery_alignment_ms)),
        "gallery_model_embedding_total_ms": float(sum(gallery_model_embedding_ms)),
        "gallery_embedding_per_identity_ms": embedding_ms,
        "accepted_references": len(gallery["names"]),
        "rejected_references": sum(row["status"] == "rejected" for row in per_reference),
        "manual_selection_references": sum(row["status"] == "manual_selection" for row in per_reference),
        "automatic_selection_references": len(gallery["automatic_references"]),
        "reference_selection_policy": reference_selection_policy,
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

    pending_index, reference = find_pending_reference(pending, pending_id)
    if selection.expected_identity is not None and reference["identity"] != selection.expected_identity:
        raise HTTPException(409, "The reference queue changed. Refresh status and try again.")

    faces = reference["faces"]

    if not 0 <= selection.face_index < len(faces):
        raise HTTPException(400, "Invalid face selection.")

    identity = reference["identity"]

    if any(name.casefold() == identity.casefold() for name in gallery["names"]):
        raise HTTPException(409, "Identity already exists.")

    try:
        embedding_started = time.perf_counter()
        embedding = request.app.state.embedder.get_embedding(
            reference["image"],
            faces[selection.face_index]
        )
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        add_embedding(gallery, identity, embedding)
        save_thumbnail(gallery, identity, reference["image"], faces[selection.face_index])
        gallery["revision"] = uuid4().hex

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
        logger.exception("Could not process selected reference face.")
        raise HTTPException(500, "Could not process selected face.") from error

    pending.pop(pending_index)


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


def find_pending_reference(pending, pending_id):
    for index, reference in enumerate(pending):
        if reference["pending_id"] == pending_id:
            return index, reference
    raise HTTPException(409, "This pending reference is stale. Refresh the gallery and try again.")


def automatic_reference(request, revision, reference_id):
    gallery = request.app.state.gallery or {}
    if gallery.get("revision") != revision:
        raise HTTPException(409, "The gallery changed. Refresh status and try again.")
    reference = gallery.get("automatic_references", {}).get(reference_id)
    if reference is None:
        raise HTTPException(409, "This automatic reference is stale. Refresh the gallery and try again.")
    return gallery, reference


@router.get("/gallery/automatic/{revision}/{reference_id}")
def get_automatic_reference(revision: str, reference_id: str, request: Request):
    _, reference = automatic_reference(request, revision, reference_id)
    return interactive_reference_preview(reference)


@router.post("/gallery/automatic/{reference_id}")
def correct_automatic_reference(reference_id: str, selection: AutomaticFaceCorrection, request: Request):
    gallery, reference = automatic_reference(request, selection.revision, reference_id)
    if not 0 <= selection.face_index < len(reference["faces"]):
        raise HTTPException(400, "Invalid face selection.")

    identity = reference["identity"]
    if selection.face_index == reference["selected_face_index"]:
        return {"message": f"Face kept for {identity}.", **get_gallery_status(request)}

    face = reference["faces"][selection.face_index]
    image = reference["image"]
    try:
        embedding = request.app.state.embedder.get_embedding(image, face)
        vector = normalized_embedding(gallery, embedding)
    except Exception as error:
        logger.exception("Could not correct selected reference face.")
        raise HTTPException(500, "Could not process selected face.") from error

    gallery["embeddings"][reference["embedding_index"]] = vector
    gallery.get("thumbnails", {}).pop(identity, None)
    save_thumbnail(gallery, identity, image, face)
    reference["selected_face_index"] = selection.face_index
    gallery.setdefault("timings", {}).setdefault("per_reference", []).append({
        "identity": identity, "status": "corrected_automatic_selection",
        "selected_face_index": selection.face_index,
    })
    gallery["revision"] = uuid4().hex
    return {"message": f"Face updated for {identity}.", **get_gallery_status(request)}


@router.post("/gallery/skip")
def skip_reference(
    action: PendingAction,
    request: Request
):
    pending = request.app.state.pending_references

    pending_id = action.pending_id

    pending_index, reference = find_pending_reference(pending, pending_id)
    if action.expected_identity is not None and reference["identity"] != action.expected_identity:
        raise HTTPException(409, "The reference queue changed. Refresh status and try again.")
    skipped = pending.pop(pending_index)

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



@router.post("/image/preview")
async def preview_image(file: UploadFile = File(...)):
    """Provide a lossless, bounded preview in a browser-compatible format."""
    try:
        rgb = decode_image(file, pixels=3_000_000, edge=2048)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise HTTPException(500, "Could not encode image preview")
        return {
            "preview": "data:image/png;base64," + base64.b64encode(encoded).decode("ascii"),
            "width": image.shape[1], "height": image.shape[0],
        }
    except ImageInputError as error:
        raise HTTPException(error.status_code, str(error)) from error
    finally:
        await file.close()


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

    filename = Path((file.filename or "").replace("\\", "/")).name or "group-photo"

    try:
        geometry = {}
        image = cv2.cvtColor(decode_image(file, geometry=geometry), cv2.COLOR_RGB2BGR)

        # Run the shared recognition pipeline.
        recognition = recognize_image(
            image,
            gallery,
            request.app.state.detector,
            request.app.state.embedder,
            request.app.state.matcher
        )

        results = recognition["results"]
        for result in results:
            if result["status"] == "error":
                logger.debug("Face processing failed: %s", result.get("error"))
                result["error"] = "The detected face could not be processed."

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
        file.file.seek(0)
        image_sha256 = hashlib.file_digest(file.file, "sha256").hexdigest()
        matcher = request.app.state.matcher

        return {
            "message": "Group photograph analysed successfully.",
            "filename": filename,
            "analysis_id": uuid4().hex,
            "image_sha256": image_sha256,
            "image_geometry": geometry,
            "gallery_identities": list(gallery["names"]),
            "recognition_rules": {
                name: getattr(matcher, name, None)
                for name in ("recognition_threshold", "margin_threshold", "possible_threshold", "possible_margin_threshold")
            },
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
            "gallery_revision": gallery.get("revision"),
            "image_width": image.shape[1],
            "image_height": image.shape[0],
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

    except ImageInputError as error:
        raise HTTPException(error.status_code, str(error)) from error
    finally:
        await file.close()



@router.get("/gallery/pending/{pending_id}")
def get_pending_reference(pending_id: str, request: Request, interactive: bool = False):
    pending = request.app.state.pending_references

    _, reference = find_pending_reference(pending, pending_id)
    if interactive:
        return interactive_reference_preview(reference)
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


def reference_preview_jpeg(image, max_edge):
    height, width = image.shape[:2]
    scale = min(1, max_edge / max(width, height))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise HTTPException(500, "Could not encode reference preview.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def interactive_reference_preview(reference):
    image = reference["image"]
    height, width = image.shape[:2]
    faces = reference["faces"]
    max_crop_edge = min(420, max(64, int((3_000_000 / max(1, len(faces))) ** 0.5)))
    candidates = []
    for index, face in enumerate(faces):
        x, y, w, h = face["bbox"]
        left, right = np.clip([x, x + w], 0, width).tolist()
        top, bottom = np.clip([y, y + h], 0, height).tolist()
        crop = None
        if right > left and bottom > top:
            margin = 0.12 * max(w, h)
            crop = reference_preview_jpeg(image[
                max(0, int(top - margin)):min(height, int(np.ceil(bottom + margin))),
                max(0, int(left - margin)):min(width, int(np.ceil(right + margin))),
            ], max_crop_edge)
        candidates.append({
            "index": index, "confidence": face["confidence"],
            "bbox": [left, top, max(0, right - left), max(0, bottom - top)], "crop": crop,
        })
    return {
        "pending_id": reference["pending_id"], "identity": reference["identity"], "filename": reference["filename"],
        "faces": len(faces), "preview": reference_preview_jpeg(image, 2200),
        "width": width, "height": height, "candidates": candidates,
    }
