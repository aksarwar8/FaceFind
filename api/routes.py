
from pathlib import Path
import base64

from face_engine.pipeline import recognize_image

import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel


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

    gallery["names"].append(identity)
    gallery["embeddings"] = np.vstack(
        [gallery["embeddings"], embedding.reshape(1, -1)]
    ).astype(np.float32)


def get_gallery_status(request: Request):
    gallery = request.app.state.gallery

    return {
        "gallery_loaded": bool(gallery and gallery["names"]),
        "people_count": len(gallery["names"]) if gallery else 0,
        "pending_references": len(request.app.state.pending_references),
        "gallery_errors": request.app.state.gallery_errors,
    }


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


@router.post("/gallery")
async def upload_gallery(
    request: Request,
    files: list[UploadFile] = File(...)
):
    detector = request.app.state.detector
    embedder = request.app.state.embedder

    gallery = {
        "names": [],
        "embeddings": np.empty((0, 128), dtype=np.float32),
        "errors": [],
        "pending": [],
    }

    pending_references = []
    seen_identities = set()

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
            continue

        if not identity:
            gallery["errors"].append(
                f"{filename}: Invalid filename"
            )
            continue

        if identity in seen_identities:
            gallery["errors"].append(
                f"{filename}: Duplicate identity"
            )
            continue

        seen_identities.add(identity)

        try:
            contents = await uploaded_file.read()

            if not contents:
                gallery["errors"].append(
                    f"{filename}: Empty file"
                )
                continue

            if len(contents) > MAX_FILE_SIZE:
                gallery["errors"].append(
                    f"{filename}: File exceeds 15 MB"
                )
                continue

            image = cv2.imdecode(
                np.frombuffer(contents, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if image is None:
                gallery["errors"].append(
                    f"{filename}: Could not read image"
                )
                continue

            faces = detector.detect(image)

            if len(faces) == 0:
                gallery["errors"].append(
                    f"{filename}: No face detected"
                )
                continue

            if len(faces) == 1:
                embedding = embedder.get_embedding(
                    image, faces[0]
                )

                add_embedding(
                    gallery, identity, embedding
                )

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

        except Exception as error:
            gallery["errors"].append(
                f"{filename}: {str(error)}"
            )

        finally:
            await uploaded_file.close()

    request.app.state.gallery = gallery
    request.app.state.pending_references = pending_references
    request.app.state.gallery_errors = gallery["errors"]

    return {
        "message": "Reference gallery processed.",
        **get_gallery_status(request),
        "pending": gallery["pending"],
    }


@router.post("/gallery/select")
def select_reference_face(
    selection: FaceSelection,
    request: Request
):
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
        embedding = request.app.state.embedder.get_embedding(
            reference["image"],
            faces[selection.face_index]
        )

        add_embedding(gallery, identity, embedding)

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
    request.app.state.gallery = None
    request.app.state.pending_references = []
    request.app.state.gallery_errors = []

    return {
        "message": "Reference gallery cleared.",
        **get_gallery_status(request),
    }



@router.post("/analyze")
async def analyze_group_photo(
    request: Request,
    file: UploadFile = File(...)
):
    # Check whether a reference gallery exists.
    gallery = request.app.state.gallery

    if gallery is None or not gallery["names"]:
        raise HTTPException(
            400,
            "Please upload a reference gallery first."
        )

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
