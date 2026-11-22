from __future__ import annotations

import base64
from pathlib import PurePosixPath
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.match import FaceMatcher


APP_TITLE = "FaceFind"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(
    title="FaceFind",
    description="Offline face detection and recognition",
    version="1.0.0",
)

# This application is designed for one local user/session at a time.
# Models are loaded lazily so importing app.py does not immediately load them.
detector: FaceDetector | None = None
embedder: FaceEmbedder | None = None
matcher: FaceMatcher | None = None

gallery: dict[str, np.ndarray] = {}
pending_references: dict[str, dict[str, Any]] = {}
gallery_errors: list[str] = []


def get_engine() -> tuple[FaceDetector, FaceEmbedder, FaceMatcher]:
    global detector, embedder, matcher

    if detector is None:
        detector = FaceDetector()
    if embedder is None:
        embedder = FaceEmbedder()
    if matcher is None:
        matcher = FaceMatcher()

    return detector, embedder, matcher


def safe_filename(filename: str | None) -> str:
    """Return only the filename, never a client-supplied directory path."""
    normalized = (filename or "").replace("\\", "/")
    return PurePosixPath(normalized).name


async def decode_upload(upload: UploadFile) -> tuple[str, np.ndarray]:
    filename = safe_filename(upload.filename)
    suffix = PurePosixPath(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {filename or 'unnamed file'}",
        )

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{filename} is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{filename} exceeds the 15 MB upload limit.",
        )

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode {filename}.",
        )

    return filename, image


def image_data_url(image: np.ndarray, max_dimension: int = 1600) -> str:
    """Encode an image for display in the local browser UI."""
    height, width = image.shape[:2]
    scale = min(max_dimension / max(height, width), 1.0)

    display_image = image
    if scale < 1.0:
        display_image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(
        ".jpg",
        display_image,
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    if not success:
        raise ValueError("Could not encode image for display.")

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + payload


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    labels: list[str] | None = None,
) -> np.ndarray:
    output = image.copy()
    labels = labels or []

    for index, detection in enumerate(detections):
        x, y, width, height = [int(value) for value in detection["bbox"]]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(output.shape[1] - 1, x + width)
        y2 = min(output.shape[0] - 1, y + height)

        color = (45, 190, 115)
        if index < len(labels) and labels[index].lower().startswith("unknown"):
            color = (65, 95, 235)
        elif index < len(labels) and labels[index].lower().startswith("possible"):
            color = (20, 165, 230)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        label = labels[index] if index < len(labels) else f"Face {index + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.55, min(output.shape[1] / 1800, 0.9))
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_top = max(0, y1 - text_height - baseline - 10)
        cv2.rectangle(
            output,
            (x1, label_top),
            (min(output.shape[1] - 1, x1 + text_width + 12), y1),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 6, max(text_height, y1 - 7)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output


def gallery_status() -> dict[str, Any]:
    return {
        "gallery_size": len(gallery),
        "identities": sorted(gallery.keys()),
        "pending_count": len(pending_references),
        "errors": list(gallery_errors),
    }


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return HTML


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "application": APP_TITLE}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return gallery_status()


@app.post("/api/gallery")
async def create_gallery(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """
    Build a gallery from a selected folder.

    A reference is accepted automatically only when exactly one face is
    detected. Multi-face images are held for explicit user selection.
    """
    global gallery_errors

    if not files:
        raise HTTPException(status_code=400, detail="Select a folder of images.")

    face_detector, face_embedder, _ = get_engine()

    gallery.clear()
    pending_references.clear()
    gallery_errors = []

    seen_identities: set[str] = set()
    processed = 0

    for upload in files:
        filename = safe_filename(upload.filename)
        if PurePosixPath(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            filename, image = await decode_upload(upload)
            identity = PurePosixPath(filename).stem.strip()

            if not identity:
                gallery_errors.append(f"{filename}: filename has no identity.")
                continue
            if identity in seen_identities:
                gallery_errors.append(
                    f"{filename}: duplicate identity name '{identity}'."
                )
                continue

            seen_identities.add(identity)
            detections = face_detector.detect(image)
            processed += 1

            if len(detections) == 1:
                gallery[identity] = face_embedder.get_embedding(
                    image, detections[0]
                )
            elif len(detections) == 0:
                gallery_errors.append(
                    f"{filename}: no face detected; reference not added."
                )
            else:
                pending_references[filename] = {
                    "identity": identity,
                    "image": image,
                    "detections": detections,
                }

        except HTTPException as exc:
            gallery_errors.append(f"{filename}: {exc.detail}")
        except Exception as exc:
            gallery_errors.append(f"{filename}: processing failed ({exc}).")

    pending = []
    for filename, item in pending_references.items():
        preview = draw_detections(item["image"], item["detections"])
        pending.append(
            {
                "filename": filename,
                "identity": item["identity"],
                "face_count": len(item["detections"]),
                "preview": image_data_url(preview, max_dimension=900),
                "faces": [
                    {
                        "index": index,
                        "confidence": round(float(face["confidence"]), 3),
                    }
                    for index, face in enumerate(item["detections"])
                ],
            }
        )

    return {
        "processed": processed,
        "accepted": len(gallery),
        "pending": pending,
        "errors": list(gallery_errors),
        "gallery": gallery_status(),
    }


@app.post("/api/gallery/select")
async def select_reference_face(request: Request) -> dict[str, Any]:
    body = await request.json()
    filename = str(body.get("filename", ""))
    face_index = body.get("face_index")

    item = pending_references.get(filename)
    if item is None:
        raise HTTPException(status_code=404, detail="Pending reference not found.")

    if not isinstance(face_index, int):
        raise HTTPException(status_code=400, detail="face_index must be an integer.")

    detections = item["detections"]
    if face_index < 0 or face_index >= len(detections):
        raise HTTPException(status_code=400, detail="Invalid face selection.")

    identity = item["identity"]
    if identity in gallery:
        raise HTTPException(
            status_code=409,
            detail=f"Identity '{identity}' is already in the gallery.",
        )

    _, face_embedder, _ = get_engine()
    try:
        embedding = face_embedder.get_embedding(
            item["image"], detections[face_index]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create embedding: {exc}",
        ) from exc

    gallery[identity] = embedding
    del pending_references[filename]

    return {
        "message": f"Added {identity} to the reference gallery.",
        "gallery": gallery_status(),
    }


@app.post("/api/gallery/skip")
async def skip_reference_face(request: Request) -> dict[str, Any]:
    body = await request.json()
    filename = str(body.get("filename", ""))

    if filename not in pending_references:
        raise HTTPException(status_code=404, detail="Pending reference not found.")

    del pending_references[filename]
    return {"message": "Reference skipped.", "gallery": gallery_status()}


@app.post("/api/analyze")
async def analyze_group_photo(file: UploadFile = File(...)) -> dict[str, Any]:
    if not gallery:
        raise HTTPException(
            status_code=400,
            detail="The reference gallery is empty. Build the gallery first.",
        )

    face_detector, face_embedder, face_matcher = get_engine()

    filename, image = await decode_upload(file)
    detections = face_detector.detect(image)

    results = []
    labels = []

    gallery_payload = {
        "names": list(gallery.keys()),
        "embeddings": np.vstack(list(gallery.values())).astype(np.float32),
    }

    for index, detection in enumerate(detections):
        try:
            embedding = face_embedder.get_embedding(image, detection)
            match_result = face_matcher.match(embedding, gallery_payload)

            identity = match_result["identity"]
            status_value = match_result["status"]

            if status_value == "recognised" and identity:
                label = identity
            elif status_value == "possible" and identity:
                label = "Possible: " + identity
            else:
                label = "Unknown"

            labels.append(label)
            results.append(
                {
                    "face_number": index + 1,
                    "identity": identity,
                    "status": status_value,
                    "label": label,
                    "best_score": round(float(match_result["best_score"]), 4),
                    "margin": round(float(match_result["margin"]), 4),
                    "bbox": detection["bbox"],
                    "confidence": round(float(detection["confidence"]), 4),
                    "candidates": match_result["candidates"],
                }
            )
        except Exception as exc:
            labels.append("Embedding error")
            results.append(
                {
                    "face_number": index + 1,
                    "identity": None,
                    "status": "error",
                    "label": "Embedding error",
                    "bbox": detection["bbox"],
                    "error": str(exc),
                }
            )

    annotated = draw_detections(image, detections, labels)

    return {
        "filename": filename,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "face_count": len(detections),
        "annotated_image": image_data_url(annotated),
        "faces": results,
        "gallery_size": len(gallery),
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FaceFind — Offline Face Recognition</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0b1020;
  --panel: #121a2d;
  --panel-2: #18233a;
  --line: #26344d;
  --text: #edf3ff;
  --muted: #9aabc5;
  --accent: #7c9cff;
  --accent-2: #a78bfa;
  --green: #43d6a0;
  --orange: #f4b860;
  --red: #ff7185;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(900px 500px at 10% -10%, rgba(82,112,210,.20), transparent 65%),
    radial-gradient(700px 400px at 100% 0%, rgba(142,91,210,.14), transparent 65%),
    var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
}
.shell { width: min(1120px, calc(100% - 36px)); margin: 0 auto; padding: 34px 0 60px; }
header { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:30px; }
.brand { display:flex; align-items:center; gap:13px; }
.logo {
  width:46px; height:46px; display:grid; place-items:center; border-radius:14px;
  background:linear-gradient(135deg,#7595ff,#a17be9); color:white; font-size:23px;
  box-shadow:0 10px 30px rgba(100,120,230,.25);
}
h1 { font-size:23px; margin:0; letter-spacing:-.6px; }
.subtitle { margin:4px 0 0; color:var(--muted); font-size:13px; }
.local-pill { border:1px solid rgba(67,214,160,.3); background:rgba(67,214,160,.08); color:#8ce8c3; border-radius:99px; padding:8px 12px; font-size:12px; white-space:nowrap; }
.hero { padding:28px; border:1px solid var(--line); border-radius:22px; background:linear-gradient(120deg,rgba(31,48,82,.85),rgba(20,27,48,.88)); margin-bottom:20px; }
.eyebrow { color:#a9baff; font-size:11px; letter-spacing:1.5px; font-weight:700; text-transform:uppercase; }
.hero h2 { font-size:clamp(24px,4vw,34px); line-height:1.12; letter-spacing:-1px; margin:12px 0 10px; max-width:650px; }
.hero p { margin:0; color:#b2bfd5; line-height:1.65; max-width:680px; font-size:14px; }
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:20px 0; }
.card { background:rgba(18,26,45,.94); border:1px solid var(--line); border-radius:18px; padding:21px; min-width:0; }
.card-head { display:flex; align-items:center; gap:12px; margin-bottom:17px; }
.step {
  width:32px; height:32px; display:grid; place-items:center; border-radius:10px;
  background:rgba(124,156,255,.13); color:#a9bdff; font-weight:700; font-size:13px;
}
.card h3 { margin:0; font-size:16px; letter-spacing:-.2px; }
.card .desc { color:var(--muted); font-size:12px; line-height:1.55; margin:-7px 0 16px; }
.drop {
  border:1px dashed #405271; border-radius:13px; padding:17px 13px; text-align:center;
  background:rgba(255,255,255,.015); cursor:pointer; transition:.18s;
}
.drop:hover { border-color:var(--accent); background:rgba(124,156,255,.06); }
.drop strong { display:block; font-size:12px; margin-bottom:5px; }
.drop span { color:var(--muted); font-size:11px; }
input[type=file] { display:none; }
.file-name { color:#c4d2e9; font-size:11px; margin:10px 0 0; overflow-wrap:anywhere; }
button {
  border:0; border-radius:11px; padding:11px 15px; font:600 12px inherit;
  font-size:12px; font-weight:650; cursor:pointer; transition:transform .15s, opacity .15s;
}
button:hover { transform:translateY(-1px); }
button:disabled { opacity:.45; cursor:not-allowed; transform:none; }
.primary { width:100%; margin-top:13px; color:#0b1225; background:linear-gradient(100deg,#9bb4ff,#b9a0ff); }
.secondary { background:#24324d; color:#dce7ff; border:1px solid #354866; }
.ghost { background:transparent; color:#bac8df; border:1px solid #34435e; }
.status {
  margin-top:13px; padding:10px 12px; border-radius:10px; background:rgba(255,255,255,.035);
  color:var(--muted); font-size:11px; line-height:1.5; min-height:36px;
}
.status.success { color:#8ce8c3; background:rgba(67,214,160,.08); }
.status.error { color:#ff9aaa; background:rgba(255,113,133,.08); }
.gallery-bar { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; padding:15px 17px; background:var(--panel-2); border:1px solid var(--line); border-radius:14px; margin:18px 0; }
.gallery-count { font-size:13px; font-weight:700; }
.gallery-count span { color:var(--green); }
.identity-list { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
.chip { border:1px solid #34435d; color:#c6d4e9; background:#1b2941; padding:5px 9px; border-radius:99px; font-size:10px; }
.pending-wrap { margin:18px 0; }
.pending-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.pending-card { border:1px solid #5b4c32; background:rgba(244,184,96,.045); padding:13px; border-radius:14px; }
.pending-card img { width:100%; border-radius:9px; display:block; margin:10px 0; background:#080d17; }
.pending-title { font-size:12px; font-weight:700; }
.pending-sub { color:var(--muted); font-size:11px; margin-top:4px; }
.face-actions { display:flex; gap:6px; flex-wrap:wrap; }
.face-actions button { padding:8px 10px; }
.section-title { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:27px 2px 12px; }
.section-title h2 { margin:0; font-size:18px; letter-spacing:-.4px; }
.section-title span { color:var(--muted); font-size:11px; }
.result-layout { display:grid; grid-template-columns:minmax(0,1.5fr) minmax(260px,.8fr); gap:15px; margin-top:17px; }
.result-image { border:1px solid var(--line); border-radius:15px; overflow:hidden; background:#080d17; min-height:180px; display:grid; place-items:center; }
.result-image img { display:block; width:100%; height:auto; }
.empty { color:#6f819d; font-size:12px; padding:35px; text-align:center; }
.result-list { display:flex; flex-direction:column; gap:8px; max-height:520px; overflow:auto; }
.result-row { display:flex; justify-content:space-between; gap:12px; padding:11px 12px; border:1px solid var(--line); border-radius:11px; background:#151f33; }
.result-row strong { display:block; font-size:12px; }
.result-row small { display:block; color:var(--muted); font-size:10px; margin-top:4px; }
.tag { align-self:flex-start; border-radius:99px; padding:5px 8px; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }
.tag.recognised { color:#8ce8c3; background:rgba(67,214,160,.12); }
.tag.possible { color:#ffd18b; background:rgba(244,184,96,.12); }
.tag.unknown,.tag.error { color:#ff9aaa; background:rgba(255,113,133,.12); }
.note { color:var(--muted); font-size:11px; line-height:1.6; margin-top:17px; }
footer { color:#62728d; font-size:10px; text-align:center; margin-top:35px; }
@media(max-width:850px) { .steps { grid-template-columns:1fr; } .result-layout { grid-template-columns:1fr; } }
@media(max-width:520px) { .shell { width:calc(100% - 22px); padding-top:20px; } header { align-items:flex-start; } .local-pill { font-size:10px; } .hero { padding:21px; } .card { padding:17px; } }
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <div class="logo">◉</div>
      <div><h1>FaceFind</h1><p class="subtitle">Offline face detection & recognition</p></div>
    </div>
    <div class="local-pill">● LOCAL PROCESSING · OFFLINE</div>
  </header>

  <section class="hero">
    <div class="eyebrow">Computer vision workspace</div>
    <h2>Find the people in your photograph.</h2>
    <p>Create a private reference gallery from individual photographs, then analyse a group photo. Face detection and matching run locally on this computer; your images are not sent to a cloud service.</p>
  </section>

  <div class="steps">
    <section class="card">
      <div class="card-head"><div class="step">01</div><h3>Build your gallery</h3></div>
      <p class="desc">Choose a folder of reference photos. Each filename becomes that person's identity.</p>
      <label class="drop" for="folderInput">
        <strong>＋ Choose reference folder</strong>
        <span>Select a folder containing JPG, PNG or WEBP images</span>
      </label>
      <input id="folderInput" type="file" webkitdirectory directory multiple accept="image/*">
      <div class="file-name" id="folderName">No folder selected</div>
      <button class="primary" id="buildButton" disabled>Build reference gallery</button>
      <div class="status" id="galleryMessage">Select a folder to begin.</div>
    </section>

    <section class="card">
      <div class="card-head"><div class="step">02</div><h3>Review references</h3></div>
      <p class="desc">If an image contains multiple faces, choose the correct face for its filename. This avoids silently assigning the wrong person.</p>
      <div class="status" id="reviewMessage">No references need review yet.</div>
      <div class="note">Images with no detected face are not added. Correct the photo and upload the folder again if needed.</div>
    </section>

    <section class="card">
      <div class="card-head"><div class="step">03</div><h3>Analyse group photo</h3></div>
      <p class="desc">Upload one group photograph. Detected faces are compared with the reference gallery.</p>
      <label class="drop" for="groupInput">
        <strong>＋ Choose group photo</strong>
        <span>Select one image to analyse</span>
      </label>
      <input id="groupInput" type="file" accept=".jpg,.jpeg,.png,.bmp,.webp">
      <div class="file-name" id="groupName">No group photo selected</div>
      <button class="primary" id="analyzeButton" disabled>Detect and identify faces</button>
      <div class="status" id="analysisMessage">Build a gallery before analysing a photo.</div>
    </section>
  </div>

  <div class="gallery-bar">
    <div><div class="gallery-count">Reference gallery: <span id="galleryCount">0 identities</span></div><div class="identity-list" id="identityList"></div></div>
    <button class="ghost" id="clearButton">Clear gallery</button>
  </div>

  <section class="pending-wrap" id="pendingSection" hidden>
    <div class="section-title"><h2>References needing your selection</h2><span>Choose the face that matches the filename</span></div>
    <div class="pending-grid" id="pendingGrid"></div>
  </section>

  <section>
    <div class="section-title"><h2>Analysis results</h2><span id="resultMeta">Waiting for a group photo</span></div>
    <div class="result-layout">
      <div class="result-image" id="resultImage"><div class="empty">Your annotated photograph will appear here.</div></div>
      <div class="card">
        <div class="card-head"><h3>Detected people</h3></div>
        <div class="result-list" id="resultList"><div class="empty">No results yet.</div></div>
      </div>
    </div>
  </section>

  <p class="note">Recognition is a similarity-based estimate, not proof of identity. Review uncertain matches and use this application only with appropriate consent.</p>
  <footer>FaceFind · Local inference with OpenCV YuNet and SFace</footer>
</div>

<script>
const folderInput = document.getElementById("folderInput");
const groupInput = document.getElementById("groupInput");
const buildButton = document.getElementById("buildButton");
const analyzeButton = document.getElementById("analyzeButton");
let selectedGroup = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, function (char) {
    return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char];
  });
}
function setStatus(id, message, type) {
  const element = document.getElementById(id);
  element.textContent = message;
  element.className = "status" + (type ? " " + type : "");
}
function updateGallery(data) {
  const size = data.gallery_size || 0;
  document.getElementById("galleryCount").textContent = size + (size === 1 ? " identity" : " identities");
  document.getElementById("identityList").innerHTML = (data.identities || []).map(function (name) {
    return '<span class="chip">' + escapeHtml(name) + '</span>';
  }).join("");
  analyzeButton.disabled = size === 0 || !selectedGroup;
}
function renderPending(items) {
  const section = document.getElementById("pendingSection");
  const grid = document.getElementById("pendingGrid");
  section.hidden = !items || items.length === 0;
  grid.innerHTML = "";
  (items || []).forEach(function (item) {
    const card = document.createElement("div");
    card.className = "pending-card";
    const faceButtons = item.faces.map(function (face) {
      return '<button class="secondary" data-face="' + face.index + '">Use face ' +
        (face.index + 1) + ' · ' + face.confidence + '</button>';
    }).join("");
    card.innerHTML =
      '<div class="pending-title">' + escapeHtml(item.identity) + '</div>' +
      '<div class="pending-sub">' + escapeHtml(item.filename) + ' · ' + item.face_count + ' faces detected</div>' +
      '<img src="' + item.preview + '" alt="Detected reference faces">' +
      '<div class="face-actions">' + faceButtons +
      '<button class="ghost" data-skip="1">Skip</button></div>';
    card.querySelectorAll("[data-face]").forEach(function (button) {
      button.addEventListener("click", function () {
        chooseReference(item.filename, Number(button.dataset.face), card);
      });
    });
    card.querySelector("[data-skip]").addEventListener("click", function () {
      skipReference(item.filename, card);
    });
    grid.appendChild(card);
  });
  setStatus("reviewMessage", items && items.length ? items.length + " reference photo(s) need a face selection." : "All detected references have been handled.", items && items.length ? "" : "success");
}
async function chooseReference(filename, faceIndex, card) {
  const response = await fetch("/api/gallery/select", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({filename:filename, face_index:faceIndex})
  });
  const data = await response.json();
  if (!response.ok) { alert(data.detail || "Could not add reference."); return; }
  card.remove();
  if (!document.getElementById("pendingGrid").children.length) {
    document.getElementById("pendingSection").hidden = true;
    setStatus("reviewMessage", "All detected references have been handled.", "success");
  }
  updateGallery(data.gallery);
}
async function skipReference(filename, card) {
  const response = await fetch("/api/gallery/skip", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({filename:filename})
  });
  const data = await response.json();
  if (!response.ok) { alert(data.detail || "Could not skip reference."); return; }
  card.remove();
  if (!document.getElementById("pendingGrid").children.length) {
    document.getElementById("pendingSection").hidden = true;
    setStatus("reviewMessage", "All detected references have been handled.", "success");
  }
  updateGallery(data.gallery);
}
folderInput.addEventListener("change", function () {
  const files = Array.from(folderInput.files || []);
  document.getElementById("folderName").textContent = files.length ? files.length + " file(s) selected" : "No folder selected";
  buildButton.disabled = files.length === 0;
});
groupInput.addEventListener("change", function () {
  selectedGroup = groupInput.files && groupInput.files[0] ? groupInput.files[0] : null;
  document.getElementById("groupName").textContent = selectedGroup ? selectedGroup.name : "No group photo selected";
  analyzeButton.disabled = !selectedGroup;
  fetch("/api/status").then(function (r) { return r.json(); }).then(updateGallery);
});
buildButton.addEventListener("click", async function () {
  const files = Array.from(folderInput.files || []);
  if (!files.length) return;
  buildButton.disabled = true;
  setStatus("galleryMessage", "Processing reference photographs…");
  const form = new FormData();
  files.forEach(function (file) {
    form.append("files", file, file.webkitRelativePath || file.name);
  });
  try {
    const response = await fetch("/api/gallery", {method:"POST", body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Gallery creation failed.");
    updateGallery(data.gallery);
    renderPending(data.pending);
    const errorCount = (data.errors || []).length;
    setStatus("galleryMessage",
      data.accepted + " reference(s) added. " + data.pending.length +
      " need review. " + errorCount + " issue(s).",
      data.accepted ? "success" : "error");
    if (errorCount) {
      setStatus("reviewMessage", (data.errors || []).slice(0, 5).join(" | "), "error");
    }
  } catch (error) {
    setStatus("galleryMessage", error.message, "error");
  } finally {
    buildButton.disabled = false;
  }
});
analyzeButton.addEventListener("click", async function () {
  if (!selectedGroup) return;
  analyzeButton.disabled = true;
  setStatus("analysisMessage", "Detecting faces and comparing embeddings…");
  const form = new FormData();
  form.append("file", selectedGroup, selectedGroup.name);
  try {
    const response = await fetch("/api/analyze", {method:"POST", body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed.");
    document.getElementById("resultImage").innerHTML =
      '<img src="' + data.annotated_image + '" alt="Annotated group photograph">';
    document.getElementById("resultMeta").textContent =
      data.face_count + " face(s) · " + data.image_width + " × " + data.image_height;
    const list = document.getElementById("resultList");
    list.innerHTML = "";
    if (!data.faces.length) {
      list.innerHTML = '<div class="empty">No faces detected in this photograph.</div>';
    }
    data.faces.forEach(function (face) {
      const row = document.createElement("div");
      row.className = "result-row";
      const score = face.best_score === undefined ? "" : "Similarity " + face.best_score;
      const margin = face.margin === undefined ? "" : " · margin " + face.margin;
      row.innerHTML =
        '<div><strong>Face ' + face.face_number + ': ' + escapeHtml(face.label) + '</strong>' +
        '<small>' + escapeHtml(score + margin) + '</small></div>' +
        '<span class="tag ' + escapeHtml(face.status) + '">' + escapeHtml(face.status) + '</span>';
      list.appendChild(row);
    });
    const recognised = data.faces.filter(function (f) { return f.status === "recognised"; }).length;
    const unknown = data.faces.filter(function (f) { return f.status === "unknown"; }).length;
    setStatus("analysisMessage",
      "Completed: " + data.face_count + " faces · " + recognised + " recognised · " + unknown + " unknown.",
      "success");
  } catch (error) {
    setStatus("analysisMessage", error.message, "error");
  } finally {
    analyzeButton.disabled = !selectedGroup;
  }
});
document.getElementById("clearButton").addEventListener("click", async function () {
  if (!confirm("Clear the current reference gallery?")) return;
  const response = await fetch("/api/gallery/clear", {method:"POST"});
  const data = await response.json();
  updateGallery(data.gallery);
  renderPending([]);
  setStatus("galleryMessage", "Gallery cleared.", "success");
  document.getElementById("resultImage").innerHTML = '<div class="empty">Your annotated photograph will appear here.</div>';
  document.getElementById("resultList").innerHTML = '<div class="empty">No results yet.</div>';
  document.getElementById("resultMeta").textContent = "Waiting for a group photo";
});
</script>
</body>
</html>"""


@app.post("/api/gallery/clear")
async def clear_gallery() -> dict[str, Any]:
    global gallery_errors
    gallery.clear()
    pending_references.clear()
    gallery_errors = []
    return {"gallery": gallery_status()}
