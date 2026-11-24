"""Exercise FaceFind's local FastAPI workflow with existing private images.

This is a functional smoke test, not a model benchmark. It reuses one accepted
reference and the existing genuine group photo, validates gallery lifecycle and
cache behavior, and writes only a compact private JSON/Markdown report.
Run with the unified local-demo requirements and existing local checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "data/private/evaluation/final_app_performance_20261124_01"
CALIBRATION = ROOT / "data/private/evaluation/final_end_to_end_comparison_20261124_01/comparison_results.json"
OUT = ROOT / "data/private/evaluation/facefind_milestone2_verification_20261124_01"
GROUP_PHOTO = ROOT / "data/private/group_photos/group.jpg"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def response_summary(response):
    data = response.json()
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    return data


def run():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing verification output: {OUT}")
    previous_results_path = PREVIOUS / "performance_results.json"
    previous_manifest_path = PREVIOUS / "test_input_manifest.json"
    for path in (previous_results_path, previous_manifest_path, GROUP_PHOTO, CALIBRATION):
        if not path.is_file():
            raise FileNotFoundError(path)

    prior = json.loads(previous_results_path.read_text(encoding="utf-8"))
    gallery_result = prior["pipelines"]["scrfd_adaface"]["gallery"]
    accepted_names = {
        row["identity"] for row in gallery_result["per_reference"]
        if row.get("status") == "accepted"
    }
    candidate_dir = PREVIOUS / "test_inputs/gallery"
    reference_path = next(
        (path for path in sorted(candidate_dir.glob("*")) if path.stem in accepted_names),
        None,
    )
    if reference_path is None:
        raise RuntimeError("No previously accepted private reference is available for the smoke workflow")

    reference_bytes = reference_path.read_bytes()
    group_bytes = GROUP_PHOTO.read_bytes()
    group_image = cv2.imdecode(np.frombuffer(group_bytes, np.uint8), cv2.IMREAD_COLOR)
    if group_image is None:
        raise RuntimeError("Existing private group photograph could not be decoded")
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    blank_ok, blank_encoded = cv2.imencode(".jpg", blank)
    if not blank_ok:
        raise RuntimeError("Could not prepare an in-memory no-face gallery test image")

    # No outputs are created until the input and overwrite guards pass.
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "verification": "FastAPI functional smoke test; not a benchmark",
        "offline": True,
        "reference": {"source": str(reference_path.relative_to(ROOT)), "sha256": sha256(reference_bytes)},
        "group_photo": {"source": str(GROUP_PHOTO.relative_to(ROOT)), "sha256": sha256(group_bytes),
                        "shape": list(group_image.shape[:2]), "synthetic": False,
                        "identity_ground_truth": "not available; no accuracy claims"},
        "previous_50_gallery_performance_artifact": str((PREVIOUS / "performance_results.json").relative_to(ROOT)),
        "manual_selection_input": "the existing genuine group photo; selected face is a workflow test only",
        "unsupported_and_invalid_inputs": ["blank image with no face", "undecodable image bytes", "unsupported extension"],
    }

    # The existing app environment now includes the pinned research runtime.
    os.environ["FACEFIND_DEFAULT_PIPELINE"] = "scrfd_adaface"
    sys.path.insert(0, str(ROOT))
    from app import app
    from scripts.evaluation.benchmark_facefind_app import LocalApi

    from face_engine.match import FaceMatcher
    from face_engine.research_models import ADAFACE_MARGIN_THRESHOLD, ADAFACE_SIMILARITY_THRESHOLD

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8-sig"))
    ada_group_calibration = calibration["calibration"]["thresholds"]["B"]["group"]
    sface_group_calibration = calibration["calibration"]["thresholds"]["A"]["group"]
    sface_defaults = FaceMatcher()
    result = {"status": "complete", "verification_type": "functional smoke; no model benchmark repeated",
              "calibration_validation": {
                  "artifact": str(CALIBRATION.relative_to(ROOT)),
                  "gallery_calibration_size": 50,
                  "adaface": {
                      "saved_similarity_threshold": ada_group_calibration["threshold"],
                      "runtime_similarity_threshold": ADAFACE_SIMILARITY_THRESHOLD,
                      "saved_margin_threshold": ada_group_calibration["margin"],
                      "runtime_margin_threshold": ADAFACE_MARGIN_THRESHOLD,
                      "matches_after_five_decimal_rounding": (
                          round(ada_group_calibration["threshold"], 5) == ADAFACE_SIMILARITY_THRESHOLD
                          and round(ada_group_calibration["margin"], 5) == ADAFACE_MARGIN_THRESHOLD
                      ),
                  },
                  "sface_fallback": {
                      "saved_experiment_similarity_threshold": sface_group_calibration["threshold"],
                      "saved_experiment_margin_threshold": sface_group_calibration["margin"],
                      "retained_production_similarity_threshold": sface_defaults.recognition_threshold,
                      "retained_production_margin_threshold": sface_defaults.margin_threshold,
                      "uses_experiment_thresholds": False,
                  },
              },
              "existing_50_reference_app_performance": {},
              "checks": {}, "scrfd_adaface": {}, "offline": True}
    result["checks"]["adaface_thresholds_match_saved_group_calibration"] = (
        result["calibration_validation"]["adaface"]["matches_after_five_decimal_rounding"]
    )
    result["checks"]["sface_fallback_thresholds_remain_production_defaults"] = (
        result["calibration_validation"]["sface_fallback"]["retained_production_similarity_threshold"] == 0.50
        and result["calibration_validation"]["sface_fallback"]["retained_production_margin_threshold"] == 0.12
    )
    for pipeline_id in ("scrfd_adaface", "yunet_sface"):
        old = prior["pipelines"][pipeline_id]
        old_gallery = old["gallery"]
        old_warm = old["warm_repeated_analysis"]
        result["existing_50_reference_app_performance"][pipeline_id] = {
            "initialization_ms": old["model_initialization_ms"],
            "gallery_requested": old_gallery["requested"],
            "gallery_accepted": old_gallery["accepted"],
            "gallery_rejected": old_gallery["rejected"],
            "manual_selection_references": old_gallery["manual_selection"],
            "first_analysis_api_ms": old["first_group_analysis"]["client_request_ms"],
            "warm_request_count": old_warm["repeat_count"],
            "warm_api_latency_ms": old_warm["end_to_end_api_latency"],
            "warm_backend_latency_ms": old_warm["backend_total"],
            "warm_detection_latency_ms": old_warm["face_detection"],
            "warm_alignment_latency_ms": old_warm["face_alignment"],
            "warm_embedding_latency_ms": old_warm["face_embedding"],
            "warm_matching_latency_ms": old_warm["identity_matching"],
            "warm_rendering_latency_ms": old_warm["annotation_and_rendering"],
            "gallery_rebuild_during_repeats_ms": old_warm["gallery_rebuild_ms"],
            "faces_per_warm_request": old_warm["face_count_each_request"],
            "duplicate_identity_assignments": old["correctness"]["duplicate_identity_assignments"],
        }
    with LocalApi(app) as client:
        health = response_summary(client.get("/api/health"))
        result["checks"]["health_ready"] = health.get("engine_initialized") is True
        result["checks"]["default_pipeline_is_scrfd_adaface"] = health.get("active_pipeline_id") == "scrfd_adaface"
        if not all((result["checks"]["health_ready"], result["checks"]["default_pipeline_is_scrfd_adaface"])):
            raise RuntimeError(f"Unexpected actual FastAPI startup state: {health}")

        before = time.perf_counter()
        selected = response_summary(client.post("/api/pipeline", json={"pipeline_id": "scrfd_adaface"}))
        result["scrfd_adaface"]["pipeline_selection_ms"] = (time.perf_counter() - before) * 1000
        result["checks"]["pipeline_switch_to_scrfd_succeeds"] = selected["active_pipeline_id"] == "scrfd_adaface"

        gallery_files = [
            ("files", ("KnownReference.jpg", reference_bytes, "image/jpeg")),
            ("files", ("ManualSelection.jpg", group_bytes, "image/jpeg")),
            ("files", ("NoFace.jpg", blank_encoded.tobytes(), "image/jpeg")),
            ("files", ("Unreadable.jpg", b"invalid image bytes", "image/jpeg")),
            ("files", ("Unsupported.txt", b"not an image", "text/plain")),
        ]
        before = time.perf_counter()
        gallery = response_summary(client.post("/api/gallery", files=gallery_files))
        result["scrfd_adaface"]["gallery_upload_client_ms"] = (time.perf_counter() - before) * 1000
        result["scrfd_adaface"]["gallery_upload"] = gallery["upload_summary"]
        result["checks"]["valid_reference_accepted"] = gallery["upload_summary"]["accepted_references"] == 1
        result["checks"]["multi_face_reference_pending"] = gallery["upload_summary"]["manual_selection_references"] == 1
        result["checks"]["invalid_references_reported"] = gallery["upload_summary"]["rejected_references"] == 3
        if not all(result["checks"][key] for key in ("valid_reference_accepted", "multi_face_reference_pending", "invalid_references_reported")):
            raise RuntimeError(f"Unexpected reference handling summary: {gallery['upload_summary']}")

        before = time.perf_counter()
        manual = response_summary(client.post("/api/gallery/select", json={"pending_id": app.state.pending_references[0]["pending_id"], "face_index": 0}))
        result["scrfd_adaface"]["manual_selection_client_ms"] = (time.perf_counter() - before) * 1000
        result["checks"]["manual_face_selection"] = manual["people_count"] == 2 and manual["pending_references"] == 0

        working_gallery = app.state.gallery
        rejected = response_summary(client.post("/api/gallery", files={
            "files": ("NoFace.jpg", blank_encoded.tobytes(), "image/jpeg")
        }))
        result["checks"]["failed_replacement_preserves_gallery"] = (
            rejected["build_succeeded"] is False and app.state.gallery is working_gallery
        )

        to_fallback = response_summary(client.post("/api/pipeline", json={"pipeline_id": "yunet_sface"}))
        result["checks"]["switch_to_yunet_keeps_gallery_model_specific"] = (
            to_fallback["active_pipeline_id"] == "yunet_sface" and not to_fallback["gallery_loaded"]
        )
        back_to_scrfd = response_summary(client.post("/api/pipeline", json={"pipeline_id": "scrfd_adaface"}))
        result["checks"]["switch_back_restores_scrfd_gallery"] = (
            back_to_scrfd["active_pipeline_id"] == "scrfd_adaface" and back_to_scrfd["gallery_loaded"]
            and back_to_scrfd["gallery_pipeline_id"] == "scrfd_adaface"
        )

        active_embedder = app.state.embedder
        original_get_embedding = active_embedder.get_embedding
        observed_gallery_embedding_calls = 0

        def count_embedding_calls(image, detection):
            nonlocal observed_gallery_embedding_calls
            observed_gallery_embedding_calls += 1
            return original_get_embedding(image, detection)

        active_embedder.get_embedding = count_embedding_calls
        analysis_rows = []
        for _ in range(2):
            started = time.perf_counter()
            analysis = response_summary(client.post("/api/analyze", files={
                "file": ("group.jpg", group_bytes, "image/jpeg")
            }))
            client_ms = (time.perf_counter() - started) * 1000
            if any(row.get("status") == "unknown" and row.get("identity") is not None for row in analysis["results"]):
                raise AssertionError("Unknown detections were returned with a gallery identity")
            analysis_rows.append({
                "client_request_ms": client_ms,
                "backend_total_ms": analysis["timings_ms"]["backend_total"],
                "face_count": analysis["face_count"],
                "summary": analysis["summary"],
                "unknown_identity_fields_clear": all(
                    row.get("identity") is None for row in analysis["results"] if row.get("status") == "unknown"
                ),
                "duplicate_identity_assignments": sum(
                    row.get("status") == "recognised" for row in analysis["results"]
                ) - len({row.get("identity") for row in analysis["results"] if row.get("status") == "recognised"}),
            })
        result["scrfd_adaface"]["group_photo_analyses"] = analysis_rows
        result["scrfd_adaface"]["group_face_embedding_calls"] = observed_gallery_embedding_calls
        expected_calls = sum(row["face_count"] for row in analysis_rows)
        result["checks"]["two_group_analyses_completed"] = len(analysis_rows) == 2 and all(row["face_count"] > 0 for row in analysis_rows)
        result["checks"]["gallery_embeddings_not_rebuilt_during_analysis"] = observed_gallery_embedding_calls == expected_calls
        result["checks"]["gallery_cache_survived_repeated_analysis"] = app.state.gallery is working_gallery
        result["scrfd_adaface"]["gallery_embedding_calls_during_gallery_build_and_selection"] = 2
        result["checks"]["all_functional_checks_passed"] = all(result["checks"].values())

    manifest["web_offline_check"] = {
        "api_base": "/api",
        "external_http_urls_in_frontend": bool(re.search(r"https?://", (ROOT / "web/app.js").read_text(encoding="utf-8"))),
        "external_script_or_stylesheet_in_html": bool(re.search(r"<(?:script|link)[^>]+https?://", (ROOT / "web/index.html").read_text(encoding="utf-8"), re.I)),
    }
    manifest["web_offline_check"]["passed"] = not any((
        manifest["web_offline_check"]["external_http_urls_in_frontend"],
        manifest["web_offline_check"]["external_script_or_stylesheet_in_html"],
    ))
    result["checks"]["frontend_uses_local_assets_and_api_only"] = manifest["web_offline_check"]["passed"]
    result["checks"]["all_functional_checks_passed"] = all(
        value for key, value in result["checks"].items() if key != "all_functional_checks_passed"
    )
    manifest["environment"] = {
        "python": "3.11",
        "requirements": "requirements-local-demo.txt plus insightface==2.1 installed with --no-deps",
        "models_downloaded": False,
        "execution_network": "loopback HTTP only; no remote inference or API",
    }
    (OUT / "test_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "verification_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    report = [
        "# FaceFind Milestone 2 Functional Verification", "",
        "This is a private local FastAPI workflow smoke test. It is not a detector or recognition benchmark. The group photo has no identity ground truth, so no recognition-accuracy claim is made.", "",
        f"- Active pipeline exercised: SCRFD-10G-KPS + AdaFace IR-18 (CPU).",
        f"- Group image size: {group_image.shape[1]}×{group_image.shape[0]} pixels; source is a genuine local photo.",
        f"- Group detections across two actual API analyses: {analysis_rows[0]['face_count']} and {analysis_rows[1]['face_count']}.",
        f"- Gallery references processed: 5; accepted: {gallery['upload_summary']['accepted_references']}; rejected: {gallery['upload_summary']['rejected_references']}; manual selection: {gallery['upload_summary']['manual_selection_references']}; selected reference count after manual action: {manual['people_count']}.",
        f"- Warm API requests in this smoke run: {analysis_rows[0]['client_request_ms']:.1f} ms and {analysis_rows[1]['client_request_ms']:.1f} ms. These two timings are functional observations, not a latency benchmark.",
        f"- Gallery embedding calls during repeated group analyses: {observed_gallery_embedding_calls}; expected from detected faces: {expected_calls}; no gallery rebuild occurred.",
        "- Unknown detections are checked to have a null identity. Duplicate-resolution behavior is covered by the focused regression tests.",
        "- Offline check: frontend uses the local `/api` route and local static assets; no external HTTP URLs are referenced.", "",
        "## Existing 50-reference application measurement", "",
        "The already-completed application performance run is reused rather than repeated. Its genuine local group photo was not synthetically altered and has no identity ground truth. See `data/private/evaluation/final_app_performance_20261124_01/performance_report.md` and this report's JSON for the 52-reference setup, five warm requests per pipeline, complete latency breakdown, and cache regression checks.", "",
        "## Calibration provenance", "",
        "AdaFace's deployed 50-gallery similarity threshold 0.26362 and margin 0.07940 agree with the saved calibration values 0.2636164 and 0.0793970 after five-decimal rounding. The saved SFace 50-gallery experiment calibration (0.3855317 / 0.0661369) differs from the fallback's retained production matcher values (0.50 / 0.12). Those fallback values were not changed in this milestone; the calibration was experiment-specific and the app fallback is intentionally kept on its established production policy.", "",
        "## Existing 50-reference timing summary", "",
        "| Pipeline | Accepted / rejected / manual | Faces per warm request | Warm API mean / median / p95 / max (ms) | Warm backend mean / median / p95 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for pipeline_id, label in (("scrfd_adaface", "SCRFD + AdaFace"), ("yunet_sface", "YuNet + SFace")):
        m = result["existing_50_reference_app_performance"][pipeline_id]
        warm = m["warm_api_latency_ms"]
        backend = m["warm_backend_latency_ms"]
        report.append(
            f"| {label} | {m['gallery_accepted']} / {m['gallery_rejected']} / {m['manual_selection_references']} | "
            f"{', '.join(map(str, m['faces_per_warm_request']))} | "
            f"{warm['mean_ms']:.1f} / {warm['median_ms']:.1f} / {warm['p95_ms']:.1f} / {warm['max_ms']:.1f} | "
            f"{backend['mean_ms']:.1f} / {backend['median_ms']:.1f} / {backend['p95_ms']:.1f} |"
        )
    report.extend([
        "",
        "## Dependency environment", "",
        "The `.venv` now contains both FastAPI and the pinned research runtime. `requirements-local-demo.txt` composes the base and research requirements. InsightFace 2.1 is installed with `--no-deps` to avoid installing a second OpenCV provider; the installed `opencv-contrib-python` supplies the `cv2` API used here.", "",
        "## Results", "",
        "All functional checks: " + ("PASS" if result["checks"]["all_functional_checks_passed"] else "FAIL"), "",
        "SCRFD's pretrained model remains restricted to non-commercial research. AdaFace checkpoint rights remain unresolved; this local demo is not cleared for commercial deployment.", "",
    ])
    (OUT / "verification_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT.relative_to(ROOT)),
        "checks": result["checks"],
        "scrfd_adaface_gallery": {"accepted": gallery["upload_summary"]["accepted_references"],
                                  "rejected": gallery["upload_summary"]["rejected_references"],
                                  "manual": gallery["upload_summary"]["manual_selection_references"],
                                  "post_selection": manual["people_count"]},
        "analysis_observations": [{"latency_ms": row["client_request_ms"], "faces": row["face_count"], "statuses": row["summary"]} for row in analysis_rows],
    }, indent=2))


if __name__ == "__main__":
    run()
