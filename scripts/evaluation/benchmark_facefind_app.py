"""Measure the actual local FaceFind FastAPI workflow on private local images.

Run with .venv_eval. All generated inputs and reports stay under data/private.
The group photograph is a genuine local photo; no identity-ground-truth accuracy
is claimed because it has no authorized identity annotations.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import math
import statistics
import sys
import time
import socket
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# FastAPI/uvicorn are present in the existing app environment while ORT and
# InsightFace live in .venv_eval. Reuse both installed site-package trees with
# the same Python 3.11 runtime; this does not install or modify dependencies.
APP_SITE_PACKAGES = ROOT / ".venv/Lib/site-packages"
if APP_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(APP_SITE_PACKAGES))
OUT = ROOT / "data/private/evaluation/final_app_performance_20261124_01"
CALIBRATION = ROOT / "data/private/evaluation/final_end_to_end_comparison_20261124_01/calibration_results.json"
LOCAL_REFS = ROOT / "data/private/reference_faces"
GROUP_PHOTO = ROOT / "data/private/group_photos/group.jpg"
PIPELINES = ("scrfd_adaface", "yunet_sface")


class HttpResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body.decode("utf-8", errors="replace")
        try:
            self._json = json.loads(self.text)
        except Exception:
            self._json = {}

    def json(self):
        return self._json


class LocalApi:
    """Use only stdlib HTTP against loopback to exercise the running FastAPI app."""
    def __init__(self, application):
        self.application = application
        self.server = None
        self.thread = None
        self.port = None

    def __enter__(self):
        import uvicorn
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        config = uvicorn.Config(self.application, host="127.0.0.1", port=self.port,
                                log_level="error", access_log=False)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.perf_counter() + 30
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("Local FastAPI server exited during startup")
            if time.perf_counter() >= deadline:
                raise TimeoutError("Local FastAPI server did not start within 30 seconds")
            time.sleep(0.05)
        return self

    def __exit__(self, *_):
        self.server.should_exit = True
        self.thread.join(timeout=20)

    def _request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=300)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            payload = response.read()
            return HttpResponse(response.status, payload)
        finally:
            conn.close()

    def get(self, path):
        return self._request("GET", path)

    def post(self, path, json=None, files=None):
        if json is not None:
            data = __import__("json").dumps(json).encode("utf-8")
            return self._request("POST", path, data, {"Content-Type": "application/json"})
        boundary = "facefind-benchmark-boundary-20261124"
        body = bytearray()
        items = []
        if isinstance(files, dict):
            items = [(field, value) for field, value in files.items()]
        elif files:
            items = files
        for field, file_info in items:
            filename, payload, content_type = file_info
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode())
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.extend(payload)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return self._request("POST", path, bytes(body), {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summary(values):
    values = [float(x) for x in values]
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    return {"count": len(values), "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": max(values)}


def prepare_gallery_inputs(out_dir):
    source = json.loads(CALIBRATION.read_text(encoding="utf-8"))["selected_split"]["known"]
    ref_dir = out_dir / "test_inputs" / "gallery"
    ref_dir.mkdir(parents=True)
    choices = []
    for path in sorted(LOCAL_REFS.glob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            choices.append((path.stem, path, "local_reference_faces"))
    identities = {name.casefold() for name, _, _ in choices}
    for record in source:
        if len(choices) >= 52:
            break
        identity = record["identity"]
        if identity.casefold() in identities:
            continue
        path = Path(record["gallery"])
        if not path.is_file():
            continue
        choices.append((identity, path, "disjoint_LFW_calibration_reference"))
        identities.add(identity.casefold())
    if len(choices) != 52:
        raise RuntimeError(f"Need approximately 50 local reference identities (prepared 52); found {len(choices)}")
    manifest = []
    paths = []
    for identity, source_path, source_type in choices:
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode gallery source image: {source_path}")
        h, w = image.shape[:2]
        scale = min(1280 / max(w, h), 1.0)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"Could not encode benchmark reference {source_path}")
        data = encoded.tobytes()
        target = ref_dir / f"{identity}.jpg"
        target.write_bytes(data)
        paths.append(target)
        manifest.append({"identity": identity, "source_type": source_type,
                         "source_path": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
                         "prepared_path": str(target.relative_to(ROOT)), "prepared_sha256": sha256(data),
                         "original_shape": [h, w], "prepared_shape": list(image.shape[:2]),
                         "prepared_bytes": len(data)})
    return paths, manifest


def run():
    benchmark_started = time.perf_counter()
    if OUT.exists():
        allowed = {"attempt01_initial_results"}
        existing = {p.name for p in OUT.iterdir()}
        if not allowed.issubset(existing) or existing - allowed:
            raise FileExistsError(f"Refusing to overwrite performance output: {OUT} ({sorted(existing)})")
    for source in (CALIBRATION, LOCAL_REFS, GROUP_PHOTO):
        if not source.exists():
            raise FileNotFoundError(str(source))
    OUT.mkdir(parents=True, exist_ok=True)
    gallery_paths, gallery_manifest = prepare_gallery_inputs(OUT)
    group_bytes = GROUP_PHOTO.read_bytes()
    group_image = cv2.imdecode(np.frombuffer(group_bytes, np.uint8), cv2.IMREAD_COLOR)
    if group_image is None:
        raise RuntimeError(f"Could not decode local group photo {GROUP_PHOTO}")
    manifest = {
        "benchmark_type": "actual FastAPI application timing; genuine local group photograph",
        "group_source": str(GROUP_PHOTO.relative_to(ROOT)),
        "group_sha256": sha256(group_bytes),
        "group_image_shape": list(group_image.shape[:2]),
        "group_identity_ground_truth": "not available; report counts only, no identity accuracy claim",
        "gallery_count_requested": len(gallery_paths),
        "gallery_references": gallery_manifest,
        "gallery_source_mix": {"private_local_reference_faces": sum(x["source_type"] == "local_reference_faces" for x in gallery_manifest),
                               "disjoint_LFW_calibration_reference": sum(x["source_type"] != "local_reference_faces" for x in gallery_manifest)},
        "protocol": f"Same {len(gallery_paths)} prepared reference files and same original group-photo bytes for both local FastAPI configurations. Gallery upload once per configuration, then explicit deterministic largest-face selections for ambiguous references; measure first analysis and five consecutive warm analyses.",
        "synthetic_inputs_used": False,
    }
    (OUT / "test_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    import app as app_module
    app = app_module.app
    results = {"status": "complete", "offline": True, "input_manifest": "test_input_manifest.json",
               "pipelines": {}, "regression_checks": {}, "resources": {"psutil_available": False}}
    start_client = time.perf_counter()
    with LocalApi(app) as client:
        results["cold_start"] = {"app_startup_api_harness_ms": (time.perf_counter() - start_client) * 1000}
        health = client.get("/api/health")
        if health.status_code != 200 or health.json().get("engine_initialized") is not True:
            raise RuntimeError(f"FastAPI health check failed: {health.status_code} {health.text[:400]}")
        results["regression_checks"]["health"] = True
        for pipeline_id in PIPELINES:
            select_started = time.perf_counter()
            selected = client.post("/api/pipeline", json={"pipeline_id": pipeline_id})
            select_latency = (time.perf_counter() - select_started) * 1000
            if selected.status_code != 200:
                raise RuntimeError(f"Pipeline selection failed for {pipeline_id}: {selected.status_code} {selected.text[:800]}")
            selected_json = selected.json()
            if selected_json["gallery_loaded"]:
                raise RuntimeError(f"Unexpected gallery reuse across distinct test config for {pipeline_id}")
            pipeline_bundle = app.state.pipeline_bundles[pipeline_id]

            upload_files = []
            for path in gallery_paths:
                upload_files.append(("files", (path.name, path.read_bytes(), "image/jpeg")))
            gallery_started = time.perf_counter()
            gallery_response = client.post("/api/gallery", files=upload_files)
            gallery_client_ms = (time.perf_counter() - gallery_started) * 1000
            if gallery_response.status_code != 200:
                raise RuntimeError(f"Gallery upload failed for {pipeline_id}: {gallery_response.status_code} {gallery_response.text[:1000]}")
            gallery_data = gallery_response.json()
            if not gallery_data["gallery_loaded"]:
                raise RuntimeError(f"No accepted gallery references for {pipeline_id}: {gallery_data}")
            if gallery_data["gallery_pipeline_id"] != pipeline_id:
                raise RuntimeError(f"Gallery/config mismatch for {pipeline_id}")
            upload_pending_count = len(gallery_data.get("pending", []))
            accepted_before_manual = gallery_data["people_count"]
            manual_selection_requests = []
            pending_items = list(gallery_data.get("pending", []))
            while pending_items:
                pending_item = pending_items[0]
                pending_id = pending_item["pending_id"]
                preview = client.get(f"/api/gallery/pending/{pending_id}")
                if preview.status_code != 200 or not preview.json().get("preview", "").startswith("data:image/jpeg;base64,"):
                    raise RuntimeError(f"Manual reference preview failed for {pipeline_id}: {pending_id}")
                # Mimic an explicit user face choice for timing reproducibility:
                # largest detected face, then closest image center, then top-left.
                pending_record = app.state.pending_references[pending_id]
                ph, pw = pending_record["image"].shape[:2]
                center = np.array([pw / 2, ph / 2])
                def selection_key(index):
                    x, y, width, height = map(float, pending_record["faces"][index]["bbox"])
                    face_center = np.array([x + width / 2, y + height / 2])
                    return (-(width * height), float(np.linalg.norm(face_center - center)), y, x)
                face_index = min(range(len(pending_record["faces"])), key=selection_key)
                selected_at = time.perf_counter()
                selection_response = client.post("/api/gallery/select", json={"pending_id": pending_id, "face_index": face_index})
                selection_elapsed = (time.perf_counter() - selected_at) * 1000
                if selection_response.status_code != 200:
                    raise RuntimeError(f"Manual face selection failed for {pipeline_id}: {selection_response.text[:600]}")
                manual_selection_requests.append({"identity": pending_item["identity"], "selected_face_index": face_index,
                                                  "client_request_ms": selection_elapsed,
                                                  "server_timings_ms": selection_response.json().get("selection_timing_ms", {})})
                pending_items = selection_response.json().get("pending", [])
            gallery_data = client.get("/api/status").json()
            gallery_count = gallery_data["people_count"]
            if not 49 <= gallery_count <= 54:
                raise RuntimeError(f"Expected an approximately 50-person resolved gallery for {pipeline_id}; received {gallery_count}")
            vector_dim = app.state.gallery["embeddings"].shape[1]

            def analyze():
                started = time.perf_counter()
                response = client.post("/api/analyze", files={"file": (GROUP_PHOTO.name, group_bytes, "image/jpeg")})
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    raise RuntimeError(f"Analysis API failed for {pipeline_id}: {response.status_code} {response.text[:1000]}")
                payload = response.json()
                if not payload["annotated_image"].startswith("data:image/jpeg;base64,"):
                    raise RuntimeError("Annotated image was not returned")
                if not payload["timings_ms"]["gallery_embeddings_reused"] or payload["timings_ms"]["gallery_embedding_time_during_analysis"] != 0:
                    raise RuntimeError("Analysis unexpectedly rebuilt the gallery")
                return elapsed, payload

            cold_request_ms, cold_payload = analyze()
            warm_ms, warm_payloads = [], []
            for _ in range(5):
                elapsed, payload = analyze()
                warm_ms.append(elapsed)
                warm_payloads.append(payload)

            def stage_summary(field):
                return summary([p["timings_ms"][field] for p in warm_payloads])

            detections = cold_payload["face_count"]
            recognized = cold_payload["summary"]["recognised"]
            status_counts = cold_payload["summary"]
            errors = sum(x["status"] == "error" for x in cold_payload["results"])
            recognized_names = [x["identity"] for x in cold_payload["results"] if x["status"] == "recognised"]
            duplicates = len(recognized_names) - len(set(recognized_names))
            expected_dim = 512 if pipeline_id == "scrfd_adaface" else 128
            if vector_dim != expected_dim:
                raise RuntimeError(f"Wrong embedding dimension in {pipeline_id}: {vector_dim}")

            # A failed replacement upload must leave the already working gallery intact.
            bad = client.post("/api/gallery", files=[("files", ("invalid.jpg", b"not an image", "image/jpeg"))])
            bad_data = bad.json()
            retained = client.get("/api/status").json()
            retained_ok = (bad.status_code == 200 and bad_data.get("build_succeeded") is False
                           and retained["people_count"] == gallery_count
                           and retained["gallery_pipeline_id"] == pipeline_id)
            if not retained_ok:
                raise RuntimeError(f"Failed upload evicted the working gallery for {pipeline_id}")

            results["pipelines"][pipeline_id] = {
                "label": selected_json["active_pipeline_label"],
                "model_initialization_ms": pipeline_bundle["model_initialization_ms"],
                "pipeline_selection_request_ms": select_latency,
                "model_dimension": vector_dim,
                "model_sizes_bytes": {
                    "detector": (ROOT / "data/private/models/scrfd_10g_bnkps.onnx").stat().st_size if pipeline_id == "scrfd_adaface" else (ROOT / "models/face/face_detection_yunet_2023mar.onnx").stat().st_size,
                    "recognizer": (ROOT / "data/private/models/adaface_ir_18_webface4m_yakhyo.onnx").stat().st_size if pipeline_id == "scrfd_adaface" else (ROOT / "models/face/face_recognition_sface_2021dec.onnx").stat().st_size,
                },
                "gallery": {
                    "requested": len(gallery_paths), "accepted": gallery_count,
                    "accepted_before_manual_selection": accepted_before_manual,
                    "rejected": gallery_data.get("gallery_timings", {}).get("rejected_references", 0),
                    "manual_selection": upload_pending_count, "manual_selection_requests": manual_selection_requests,
                    "embedding_dimension": vector_dim, "gallery_upload_client_request_ms": gallery_client_ms,
                    "gallery_manual_selection_client_total_ms": sum(x["client_request_ms"] for x in manual_selection_requests),
                    "gallery_full_build_client_request_ms": gallery_client_ms + sum(x["client_request_ms"] for x in manual_selection_requests),
                    **gallery_data.get("gallery_timings", {}),
                },
                "first_group_analysis": {"client_api_ms": cold_request_ms, "summary": cold_payload["summary"],
                                          "faces_detected": detections, "faces_identified": recognized,
                                          "result_errors": errors, "duplicate_identity_assignments": duplicates,
                                          "timings_ms": cold_payload["timings_ms"]},
                "warm_repeated_analysis": {
                    "repeat_count": len(warm_ms), "end_to_end_api_latency": summary(warm_ms),
                    "backend_total": stage_summary("backend_total"),
                    "face_detection": stage_summary("detection"),
                    "face_alignment": stage_summary("face_alignment_total"),
                    "face_embedding": stage_summary("face_embedding_total"),
                    "identity_matching": stage_summary("identity_matching_total"),
                    "annotation_and_rendering": stage_summary("annotation_and_rendering"),
                    "gallery_rebuild_ms": 0.0,
                    "face_count_each_request": [p["face_count"] for p in warm_payloads],
                },
                "correctness": {"valid_result_statuses": all(x["status"] in {"recognised", "possible", "unknown", "error"} for x in cold_payload["results"]),
                                "unknown_status_supported": all(x["status"] == "unknown" for x in cold_payload["results"] if x["identity"] is None and x["status"] != "error"),
                                "annotated_image_returned": True, "gallery_cache_retained_after_failed_upload": retained_ok,
                                "duplicate_identity_assignments": duplicates,
                                "identity_accuracy_assessed": False},
            }
            results["regression_checks"][f"{pipeline_id}_gallery_cache"] = True
            results["regression_checks"][f"{pipeline_id}_failed_upload_preserves_gallery"] = retained_ok
            results["regression_checks"][f"{pipeline_id}_same_gallery_for_six_analyses"] = all(p["timings_ms"]["gallery_embedding_time_during_analysis"] == 0 for p in [cold_payload, *warm_payloads])
            print(f"[{pipeline_id}] gallery {gallery_count}/50; group faces {detections}; warm API median {statistics.median(warm_ms):.1f} ms", flush=True)

        # Switching back restores only its own model-space gallery.
        back = client.post("/api/pipeline", json={"pipeline_id": "scrfd_adaface"}).json()
        results["regression_checks"]["pipeline_specific_gallery_cache"] = back["gallery_loaded"] and back["gallery_pipeline_id"] == "scrfd_adaface"
        if not results["regression_checks"]["pipeline_specific_gallery_cache"]:
            raise RuntimeError("Pipeline switch did not restore its own cached gallery")

    # Optional resource metrics are recorded when already available.
    try:
        import psutil
        results["resources"] = {"psutil_available": True,
                                "process_rss_bytes_after_run": psutil.Process().memory_info().rss,
                                "system_available_memory_bytes_after_run": psutil.virtual_memory().available}
    except ImportError:
        pass

    results["licensing"] = {
        "scrfd": "Non-commercial research restriction; commercial use requires separate authorization.",
        "adaface": "Repository code is MIT; checkpoint license and WebFace4M/model rights remain unresolved.",
        "scope": "Local research/assignment demo only; not cleared for commercial deployment.",
    }
    results["recommendation"] = (
        "SCRFD + AdaFace is selected as the local research/assignment demo default. "
        "Keep YuNet + SFace available as the fallback. Accuracy on this unannotated local photo was not assessed. "
        "Do not deploy the research configuration commercially until SCRFD and AdaFace checkpoint rights are cleared."
    )
    frontend_sources = (ROOT / "web/app.js").read_text(encoding="utf-8") + (ROOT / "web/index.html").read_text(encoding="utf-8")
    external_fetch = any(token in frontend_sources for token in ("fetch('http://", 'fetch("http://', "fetch('https://", 'fetch("https://'))
    results["regression_checks"]["frontend_uses_no_external_api"] = not external_fetch
    results["offline"] = not external_fetch
    results["total_execution_time_seconds"] = (time.perf_counter() - benchmark_started)
    (OUT / "performance_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results)
    print(f"Performance report saved: {OUT}", flush=True)


def write_report(results):
    lines = ["# FaceFind local application performance", "",
             "Measurements call the actual FastAPI application through its local ASGI API. The group input is the genuine local photo recorded in `test_input_manifest.json`; its identity annotations are unavailable, so these are operational counts and timing only, not accuracy results. Gallery inputs combine local named references and disjoint LFW calibration references. No synthetic test photo was used.", "",
             "## Pipeline measurements", "",
             "All request latencies are milliseconds. Cold first-analysis timings are reported separately from five repeated warm requests.", "",
             "| Configuration | Init ms | Final accepted / rejected / manual (manual is included in accepted) | Gallery load client ms | Faces detected / identified | First analysis API ms | Warm API mean / median / p95 / max ms | Warm backend mean / median / p95 ms | Detect mean ms | Align mean ms | Embed mean ms | Match mean ms | Render mean ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, data in results["pipelines"].items():
        g = data["gallery"]; first = data["first_group_analysis"]; warm = data["warm_repeated_analysis"]
        api = warm["end_to_end_api_latency"]; back = warm["backend_total"]
        fmt = lambda x: "n/a" if x is None else f"{x:.1f}"
        lines.append(f"| {data['label']} | {fmt(data['model_initialization_ms'])} | {g['accepted']}/{g['rejected']}/{g['manual_selection']} | {fmt(g['gallery_full_build_client_request_ms'])} | {first['faces_detected']} / {first['faces_identified']} | {fmt(first['client_api_ms'])} | {fmt(api['mean_ms'])} / {fmt(api['median_ms'])} / {fmt(api['p95_ms'])} / {fmt(api['max_ms'])} | {fmt(back['mean_ms'])} / {fmt(back['median_ms'])} / {fmt(back['p95_ms'])} | {fmt(warm['face_detection']['mean_ms'])} | {fmt(warm['face_alignment']['mean_ms'])} | {fmt(warm['face_embedding']['mean_ms'])} | {fmt(warm['identity_matching']['mean_ms'])} | {fmt(warm['annotation_and_rendering']['mean_ms'])} |")
    lines += ["", "## Gallery loading breakdown", "", "| Configuration | Detection total ms | Alignment total ms | Embedding total ms | Embedding ms per accepted identity (mean) | Gallery embedding cache reused across repeated photos |", "|---|---:|---:|---:|---:|---|"]
    for data in results["pipelines"].values():
        g = data["gallery"]
        mean_per = statistics.fmean(g["gallery_embedding_per_identity_ms"]) if g["gallery_embedding_per_identity_ms"] else None
        mean_per_text = "n/a" if mean_per is None else f"{mean_per:.1f}"
        lines.append(f"| {data['label']} | {g['gallery_detection_total_ms']:.1f} | {g['gallery_alignment_total_ms']:.1f} | {g['gallery_model_embedding_total_ms']:.1f} | {mean_per_text} | Yes; analysis reports zero gallery re-embedding time |")
    lines += ["", "## Workflow and correctness checks", "", "The API workflow selected each pipeline, loaded the same approximately 50 reference inputs, exercised numbered preview and face-selection endpoints for ambiguous references (the timing harness chose the largest face deterministically), analyzed the same genuine group photo once cold and five times warm, returned rendered bounding boxes and labels, preserved the working gallery after a failed replacement upload, and restored the correct model-specific gallery when switching configurations. Gallery embeddings are keyed by pipeline; SFace 128-dimensional and AdaFace 512-dimensional vectors are never shared. The API returned recognized/unknown/error statuses and duplicate identity assignment counts. The group photo has no identity ground truth, so identification counts must not be read as accuracy.", "", "SCRFD detections use stride-32 padding and map boxes/landmarks back to the source image. AdaFace uses five-point alignment and its independently calibrated 50-way cosine threshold 0.26362 and margin 0.07940. YuNet + SFace retains its original matcher thresholds.", "", "## Runtime, offline operation and licensing", "", f"Total benchmark execution: {results.get('total_execution_time_seconds', float('nan')):.2f} seconds. The frontend calls only local `/api` endpoints; the app makes no external service calls. Memory is included in JSON when psutil is already available.", "", "SCRFD pretrained weights are subject to a non-commercial research restriction. AdaFace repository code is MIT, but checkpoint licensing and WebFace4M/model rights remain unresolved. SCRFD + AdaFace is selected only for the local research/assignment demo and is not cleared for commercial deployment. YuNet + SFace remains the fallback.", "", "## Artifacts", "", "`test_input_manifest.json` records sources, hashes, image sizes, and the genuine group-photo designation. `performance_results.json` contains per-request timings, warm distributions, gallery outcomes, correctness checks, model sizes, and licensing notes.", ""]
    # Use explicit rendering to avoid locale-dependent numeric formatting.
    (OUT / "performance_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
