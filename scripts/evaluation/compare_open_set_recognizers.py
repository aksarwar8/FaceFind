"""Manual CPU comparison of SFace, antelopev2 ArcFace, and AdaFace on LFW.

The test identities/images are read verbatim from the existing open-set v2
manifest. Recognition thresholds are calibrated only on disjoint identities
drawn from the remaining LFW identity folders. Running this script writes a
new output directory; it refuses to reuse an existing directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from face_engine.detector import FaceDetector
from face_engine.embedder import MODEL_PATH as SFACE_PATH

DEFAULT_MANIFEST = ROOT / "data/private/evaluation/open_set_run_20261123_v2/reproducibility_manifest.json"
DEFAULT_IMAGES = ROOT / "data/benchmarks/lfw/lfw"
ARCFACE_PATH = ROOT / "data/private/models/antelopev2_extract/antelopev2/glintr100.onnx"
ADAFACE_PATH = ROOT / "data/private/models/adaface_ir_18_webface4m_yakhyo.onnx"
SUPPORTED = {".jpg", ".jpeg", ".png"}
ALIGN_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                      [56.0252, 71.7366], [41.5493, 92.3655],
                      [70.7299, 92.2041]], dtype=np.float32)


def norm_crop(image: np.ndarray, landmarks: list[float]) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    transform = SimilarityTransform()
    if not transform.estimate(points, ALIGN_DST):
        raise ValueError("five-point face alignment failed")
    matrix = transform.params[:2, :].astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("five-point face alignment failed")
    return cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding has invalid norm")
    return vector / norm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Encoder:
    def __init__(self, name: str, model_path: Path):
        self.name, self.path = name, model_path
        if name == "SFace":
            self.cv_recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")
            self.session = None
        else:
            self.cv_recognizer = None
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            if self.session.get_providers()[0] != "CPUExecutionProvider":
                raise RuntimeError(f"{name}: CPUExecutionProvider was not selected")
            self.input = self.session.get_inputs()[0]
            self.output = self.session.get_outputs()[0]
            if len(self.input.shape) != 4 or self.input.shape[1:] != [3, 112, 112]:
                raise ValueError(f"{name}: unsupported input contract {self.input.shape}")
            if len(self.output.shape) != 2 or self.output.shape[1] not in (128, 512):
                raise ValueError(f"{name}: unsupported output contract {self.output.shape}")
            self.dimension = int(self.output.shape[1])

    def embed(self, image: np.ndarray, face: dict) -> tuple[np.ndarray, np.ndarray]:
        if self.name == "SFace":
            row = np.asarray([*face["bbox"], *face["landmarks"], face["confidence"]], dtype=np.float32)
            aligned = self.cv_recognizer.alignCrop(image, row)
            vector = self.cv_recognizer.feature(aligned).flatten()
        else:
            aligned = norm_crop(image, face["landmarks"])
            swap_rb = self.name == "ArcFace"
            blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112),
                                         (127.5, 127.5, 127.5), swapRB=swap_rb)
            vector = self.session.run([self.output.name], {self.input.name: blob})[0].reshape(-1)
        return unit(vector), aligned


def image_paths(manifest: dict) -> set[Path]:
    paths: set[Path] = set()
    for item in manifest["known"].values():
        paths.add(Path(item["gallery_image"]).resolve())
        paths.update(Path(p).resolve() for p in item["probe_images"])
    for values in manifest["unknown"].values():
        paths.update(Path(p).resolve() for p in values)
    return paths


def calibration_split(image_root: Path, excluded: set[str], seed: int,
                      known_count: int = 200, unknown_count: int = 40) -> tuple[list, list]:
    candidates = []
    for folder in sorted(image_root.iterdir()):
        if not folder.is_dir() or folder.name in excluded:
            continue
        images = sorted(p.resolve() for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in SUPPORTED)
        if len(images) >= 2:
            candidates.append((folder.name, images))
    random.Random(seed).shuffle(candidates)
    if len(candidates) < known_count + unknown_count:
        raise RuntimeError("Not enough LFW identities outside the fixed test manifest for calibration")
    known, unknown = candidates[:known_count], candidates[known_count:known_count + unknown_count]
    return ([{"identity": n, "gallery": ps[0], "probes": ps[1:]} for n, ps in known],
            [{"identity": n, "probes": ps} for n, ps in unknown])


def cosine_scores(gallery: dict[str, np.ndarray], probe: np.ndarray) -> tuple[str, float, float]:
    names = list(gallery)
    matrix = np.vstack([gallery[n] for n in names])
    scores = matrix @ probe
    order = np.argsort(scores)[::-1]
    return names[int(order[0])], float(scores[order[0]]), float(scores[order[1]]) if len(order) > 1 else -1.0


def choose_threshold(known_scores: list[float], unknown_scores: list[float]) -> float:
    if not known_scores or not unknown_scores:
        raise RuntimeError("Threshold calibration requires known and unknown calibration scores")
    values = sorted(set(known_scores + unknown_scores))
    candidates = [values[0] - 1e-6, *[(a + b) / 2 for a, b in zip(values, values[1:])], values[-1] + 1e-6]
    # Maximize balanced acceptance accuracy; ties select the higher, conservative threshold.
    return max(candidates, key=lambda t: ((sum(s >= t for s in known_scores) / len(known_scores) +
                                           sum(s < t for s in unknown_scores) / len(unknown_scores)) / 2, t))


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "count": 0}
    return {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "count": len(values)}


def verification_metrics(genuine: list[float], impostor: list[float]) -> dict:
    """Calculate ROC AUC and interpolated EER from genuine/impostor cosine scores."""
    if not genuine or not impostor:
        return {"genuine_pairs": len(genuine), "impostor_pairs": len(impostor),
                "eer": None, "roc_auc": None}
    scores = np.asarray(genuine + impostor, dtype=np.float64)
    labels = np.concatenate((np.ones(len(genuine), dtype=np.int8),
                             np.zeros(len(impostor), dtype=np.int8)))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = fp = 0
    tpr, fpr = [0.0], [0.0]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        tp += int(sorted_labels[i:j].sum())
        fp += int((j - i) - sorted_labels[i:j].sum())
        tpr.append(tp / len(genuine))
        fpr.append(fp / len(impostor))
        i = j
    tpr_a, fpr_a = np.asarray(tpr), np.asarray(fpr)
    fnr_a = 1.0 - tpr_a
    diff = fpr_a - fnr_a
    crossing = np.flatnonzero(diff >= 0)
    if not len(crossing):
        eer = float((fpr_a[-1] + fnr_a[-1]) / 2)
    else:
        k = int(crossing[0])
        if k == 0:
            eer = float((fpr_a[0] + fnr_a[0]) / 2)
        else:
            d0, d1 = diff[k - 1], diff[k]
            ratio = 0.0 if d1 == d0 else -d0 / (d1 - d0)
            eer = float((fpr_a[k - 1] + ratio * (fpr_a[k] - fpr_a[k - 1]) +
                         fnr_a[k - 1] + ratio * (fnr_a[k] - fnr_a[k - 1])) / 2)
    return {"genuine_pairs": len(genuine), "impostor_pairs": len(impostor),
            "eer": eer, "roc_auc": float(np.trapz(tpr_a, fpr_a))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Must not already exist; default is a new UTC timestamped evaluation folder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest_path, image_root = args.manifest.resolve(), args.images_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else ROOT / "data/private/evaluation" / (
        "recognition_comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if not manifest_path.is_file() or not image_root.is_dir():
        raise FileNotFoundError(f"Missing manifest or LFW image root: {manifest_path} / {image_root}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {out}")
    models = {"SFace": SFACE_PATH, "ArcFace": ARCFACE_PATH, "AdaFace": ADAFACE_PATH}
    missing = [str(path) for path in models.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model(s): " + ", ".join(missing))
    test = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(test["known"]) != 200 or len(test["unknown"]) != 100:
        raise ValueError(f"Expected fixed 200/100 test identities; got {len(test['known'])}/{len(test['unknown'])}")
    excluded = set(test["known"]) | set(test["unknown"])
    cal_known, cal_unknown = calibration_split(image_root, excluded, args.seed)
    calibration_ids = {x["identity"] for x in cal_known + cal_unknown}
    if calibration_ids & excluded or len(calibration_ids) != len(cal_known) + len(cal_unknown):
        raise AssertionError("Calibration identities overlap test identities or each other")
    test_gallery = [(ident, Path(item["gallery_image"]).resolve()) for ident, item in test["known"].items()]
    test_probes = [("known", ident, Path(path).resolve()) for ident, item in test["known"].items()
                   for path in item["probe_images"]]
    test_probes += [("unknown", ident, Path(path).resolve()) for ident, paths in test["unknown"].items()
                    for path in paths]
    if len({str(p) for _, _, p in test_probes}) != len(test_probes):
        raise ValueError("Fixed v2 test probe list contains duplicate paths")
    test_path_digest = hashlib.sha256("\n".join(sorted(str(p) for _, _, p in test_probes)).encode()).hexdigest()
    gallery_path_digest = hashlib.sha256("\n".join(sorted(str(p) for _, p in test_gallery)).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    (out / "cache").mkdir()
    detector = FaceDetector(score_threshold=0.5, nms_threshold=0.3, max_dimension=1280)
    detections_cache: dict[str, tuple[list | None, dict | None]] = {}
    calibrations: dict[str, dict] = {}

    report = {"protocol": "LFW open-set identification; fixed v2 test manifest; independently calibrated cosine thresholds",
              "manifest": str(manifest_path), "seed": args.seed,
              "test_identity_counts": {"known": len(test["known"]), "unknown": len(test["unknown"])},
              "test_probe_count": len(test_probes), "test_probe_path_sha256": test_path_digest,
              "test_gallery_path_sha256": gallery_path_digest,
              "calibration_identity_counts": {"known": len(cal_known), "unknown": len(cal_unknown)},
              "calibration_identity_overlap_with_test": sorted(calibration_ids & excluded),
              "runtime": {"python": platform.python_version(), "opencv": cv2.__version__,
                          "onnxruntime": ort.__version__, "inference_provider": "CPUExecutionProvider",
                          "no_dependency_install": True},
              "detector": {"name": "FaceFind YuNet", "score_threshold": 0.5,
                           "nms_threshold": 0.3, "max_dimension": 1280,
                           "face_policy": "evaluate only images with exactly one detected face"},
              "model_sources_and_licenses": {
                  "SFace": {"source": "OpenCV Zoo face_recognition_sface_2021dec.onnx; MobileFaceNet/SFace",
                            "url": "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface",
                            "license": "OpenCV Zoo model folder Apache-2.0; underlying training-data/weight provenance needs separate review for deployment"},
                  "ArcFace": {"source": "InsightFace antelopev2 package, glintr100.onnx; ResNet100/Glint360K",
                              "url": "https://github.com/deepinsight/insightface/tree/master/model_zoo",
                              "license": "InsightFace pretrained antelopev2 is restricted to non-commercial research unless separately licensed"},
                  "AdaFace": {"source": "yakhyo/adaface-onnx release adaface_ir_18.onnx; IR-18/WebFace4M exported from official AdaFace weights",
                              "url": "https://github.com/yakhyo/adaface-onnx/releases/tag/weights",
                              "license": "Repository code is MIT; release does not state a separate weight license or checksum, and WebFace4M training-data/model rights remain unresolved; research-only evaluation pending rights review"}},
              "models": {}, "created_utc": datetime.now(timezone.utc).isoformat()}

    calibration_manifest = {"known": [{"identity": x["identity"], "gallery": str(x["gallery"]),
                                       "probes": list(map(str, x["probes"]))} for x in cal_known],
                           "unknown": [{"identity": x["identity"], "probes": list(map(str, x["probes"]))}
                                       for x in cal_unknown]}
    (out / "calibration_split.json").write_text(json.dumps(calibration_manifest, indent=2), encoding="utf-8")

    def embed_one(encoder: Encoder, cache_dir: Path, embeddings: dict, latencies: dict,
                  path: Path) -> tuple[np.ndarray | None, dict | None]:
        key = str(path.resolve())
        if key in embeddings:
            return embeddings[key], None
        image = cv2.imread(key)
        if image is None:
            return None, {"failure": "image_read_failure", "detail": "cv2.imread returned None"}
        if key not in detections_cache:
            try:
                detections_cache[key] = (detector.detect(image), None)
            except Exception as exc:
                detections_cache[key] = (None, {"failure": "detection_failure", "detail": str(exc)})
        faces, detection_error = detections_cache[key]
        if detection_error:
            return None, detection_error
        if not faces:
            return None, {"failure": "no_face_detected"}
        if len(faces) != 1:
            return None, {"failure": "multiple_faces_detected", "face_count": len(faces)}
        try:
            started = time.perf_counter()
            vector, aligned = encoder.embed(image, faces[0])
            latencies[key] = (time.perf_counter() - started) * 1000
        except Exception as exc:
            return None, {"failure": "embedding_failure", "detail": str(exc)}
        embeddings[key] = vector
        np.savez_compressed(cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".npz"),
                            embedding=vector, aligned_bgr=aligned)
        return vector, None

    # Calibration phase for all models completes and is frozen before any test probe is scored.
    for name, model_path in models.items():
        print(f"[{name}] calibration phase: initializing CPU encoder", flush=True)
        encoder = Encoder(name, model_path)
        cache_dir = out / "cache" / name.lower()
        cache_dir.mkdir()
        embeddings, latencies = {}, {}
        cal_gallery, failures = {}, []
        for item in cal_known:
            emb, failure = embed_one(encoder, cache_dir, embeddings, latencies, item["gallery"])
            if failure:
                failures.append({"role": "known_gallery", "identity": item["identity"], **failure})
            else:
                cal_gallery[item["identity"]] = emb
        genuine, impostor = [], []
        for kind, cohort in (("known", cal_known), ("unknown", cal_unknown)):
            for item in cohort:
                for path in item["probes"]:
                    emb, failure = embed_one(encoder, cache_dir, embeddings, latencies, path)
                    if failure:
                        failures.append({"role": f"{kind}_probe", "identity": item["identity"],
                                         "image": str(path), **failure})
                    elif kind == "known" and item["identity"] in cal_gallery:
                        genuine.append(float(cal_gallery[item["identity"]] @ emb))
                    elif kind == "unknown" and cal_gallery:
                        impostor.append(cosine_scores(cal_gallery, emb)[1])
        threshold = choose_threshold(genuine, impostor)
        calibrations[name] = {"threshold": threshold, "genuine": genuine, "impostor": impostor,
                              "gallery_count": len(cal_gallery), "failure_count": len(failures),
                              "failures": failures}
        print(f"[{name}] calibration complete: threshold={threshold:.6f}; "
              f"genuine={len(genuine)} impostor={len(impostor)} failures={len(failures)}", flush=True)
        np.savez_compressed(cache_dir / "calibration_embeddings.npz",
                            paths=np.asarray(list(embeddings)), embeddings=np.vstack(list(embeddings.values())))
        del encoder, embeddings, latencies

    frozen = {name: {"threshold": data["threshold"], "calibration_gallery_count": data["gallery_count"],
                     "genuine_scores": len(data["genuine"]), "unknown_max_scores": len(data["impostor"]),
                     "calibration_failures": data["failure_count"]}
              for name, data in calibrations.items()}
    (out / "frozen_thresholds.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("All model-specific thresholds frozen; beginning fixed test evaluation.", flush=True)

    for name, model_path in models.items():
        print(f"[{name}] test phase: initializing CPU encoder", flush=True)
        encoder = Encoder(name, model_path)
        cache_dir = out / "cache" / name.lower()
        embeddings, latencies = {}, {}
        gallery, gallery_failures = {}, []
        for ident, path in test_gallery:
            emb, failure = embed_one(encoder, cache_dir, embeddings, latencies, path)
            if failure:
                gallery_failures.append({"identity": ident, "image": str(path), **failure})
            else:
                gallery[ident] = emb
        if gallery_failures or len(gallery) != 200:
            raise RuntimeError(f"{name}: could not build identical 200-identity test gallery; "
                               f"usable={len(gallery)} failures={gallery_failures[:3]}")
        rows, genuine_test, impostor_test = [], [], []
        prediction_path = out / f"predictions_{name.lower()}.jsonl"
        with prediction_path.open("x", encoding="utf-8") as stream:
            for ix, (kind, truth, path) in enumerate(test_probes, 1):
                emb, failure = embed_one(encoder, cache_dir, embeddings, latencies, path)
                if failure:
                    row = {"split": kind, "true_identity": truth, "image": str(path),
                           "status": "failed", **failure}
                else:
                    predicted, score, second = cosine_scores(gallery, emb)
                    accepted = score >= calibrations[name]["threshold"]
                    row = {"split": kind, "true_identity": truth, "image": str(path),
                           "status": "evaluated", "rank1_identity": predicted,
                           "identity": predicted if accepted else "unknown", "accepted": accepted,
                           "best_score": score, "second_score": second}
                    if kind == "known":
                        genuine_test.append(float(gallery[truth] @ emb))
                        impostor_test.extend(float(v) for ident, ref in gallery.items() if ident != truth
                                             for v in [float(ref @ emb)])
                    else:
                        impostor_test.extend(float(ref @ emb) for ref in gallery.values())
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                if ix % 50 == 0 or ix == len(test_probes):
                    print(f"[{name}] test probes {ix}/{len(test_probes)}", flush=True)

        known_all = [r for r in rows if r["split"] == "known"]
        unknown_all = [r for r in rows if r["split"] == "unknown"]
        known_ok = [r for r in known_all if r["status"] == "evaluated"]
        unknown_ok = [r for r in unknown_all if r["status"] == "evaluated"]
        correct_rank1 = sum(r["rank1_identity"] == r["true_identity"] for r in known_ok)
        correct_accept = sum(r["accepted"] and r["rank1_identity"] == r["true_identity"] for r in known_ok)
        unknown_false_accept = sum(r["accepted"] for r in unknown_ok)
        failure_counts = {}
        for row in rows:
            if row["status"] == "failed":
                reason = row["failure"]
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        probe_latency = [latencies[str(path)] for _, _, path in test_probes if str(path) in latencies]
        total_known, total_unknown = len(known_all), len(unknown_all)
        report["models"][name] = {
            "model_path": str(model_path), "model_bytes": model_path.stat().st_size,
            "embedding_dimension": int(encoder.dimension if name != "SFace" else 128),
            "sha256": sha256_file(model_path),
            "input_contract": ("SFace FaceRecognizerSF.alignCrop/feature, BGR" if name == "SFace"
                               else "112x112 NCHW float32; 5-point similarity alignment; RGB; (x-127.5)/127.5"
                               if name == "ArcFace"
                               else "112x112 NCHW float32; 5-point similarity alignment; BGR; (x-127.5)/127.5"),
            "calibration": {"threshold": calibrations[name]["threshold"],
                            "gallery_identities": calibrations[name]["gallery_count"],
                            "genuine_pairs": len(calibrations[name]["genuine"]),
                            "unknown_max_scores": len(calibrations[name]["impostor"]),
                            "failures": calibrations[name]["failure_count"]},
            "test": {"probe_count": len(rows), "evaluated": len(known_ok) + len(unknown_ok),
                     "detection_and_embedding_failures": failure_counts,
                     "known": {"probes": total_known, "evaluated": len(known_ok),
                               "conditional_rank1_accuracy": correct_rank1 / len(known_ok) if known_ok else None,
                               "end_to_end_rank1_accuracy": correct_rank1 / total_known if total_known else None,
                               "correct_identification_and_acceptance": correct_accept,
                               "conditional_correct_acceptance_rate": correct_accept / len(known_ok) if known_ok else None,
                               "end_to_end_correct_acceptance_rate": correct_accept / total_known if total_known else None,
                               "conditional_fnir": 1 - correct_accept / len(known_ok) if known_ok else None,
                               "end_to_end_fnir": 1 - correct_accept / total_known if total_known else None},
                     "unknown": {"probes": total_unknown, "evaluated": len(unknown_ok),
                                 "conditional_rejection_rate": (len(unknown_ok) - unknown_false_accept) / len(unknown_ok) if unknown_ok else None,
                                 "end_to_end_rejection_rate": (total_unknown - unknown_false_accept) / total_unknown if total_unknown else None,
                                 "conditional_fpir": unknown_false_accept / len(unknown_ok) if unknown_ok else None,
                                 "end_to_end_false_identification_rate": unknown_false_accept / total_unknown if total_unknown else None}},
            "verification": verification_metrics(genuine_test, impostor_test),
            "embedding_latency_ms": latency_summary(probe_latency),
            "prediction_file": prediction_path.name,
            "cache_entries": len(embeddings)}
        np.savez_compressed(cache_dir / "test_embeddings.npz",
                            paths=np.asarray(list(embeddings)), embeddings=np.vstack(list(embeddings.values())))
        del encoder, embeddings, latencies

    report["test_image_identity_hash"] = test_path_digest
    report["calibration_identity_overlap_with_test"] = sorted(calibration_ids & excluded)
    (out / "comparison_results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# SFace vs ArcFace vs AdaFace — LFW open-set comparison", "",
             f"Test manifest: `{manifest_path}`  ",
             f"Test identities: {len(test['known'])} known, {len(test['unknown'])} unknown; {len(test_probes)} probes  ",
             f"Calibration: {len(cal_known)} known and {len(cal_unknown)} unknown identities, disjoint from test; usable calibration gallery: {next(iter(report['models'].values()))['calibration']['gallery_identities']}/200.  ",
             f"Gallery path SHA-256: `{gallery_path_digest}`  ",
             f"Probe path SHA-256: `{test_path_digest}`", "",
             "Each model threshold was selected independently on the calibration split and frozen before any test scoring. All models used the same fixed YuNet detections, 200-identity test gallery and probe list. Rank-1 is the top cosine identity before thresholding; correct acceptance requires the correct identity and score at or above that model's threshold. Conditional rates use evaluated probes only; end-to-end rates use all probes, treating failures as known misses and non-false-accepts for unknowns. Failed detection/embedding probes are separately counted.", "",
             "| Model | Rank-1 conditional | Rank-1 end-to-end | Correct accept conditional | Correct accept end-to-end | Unknown reject conditional | FPIR conditional | FPIR end-to-end | FNIR conditional | FNIR end-to-end | EER | ROC-AUC | Embed mean / median / p95 ms | Size MB | Failures |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data in report["models"].items():
        k, u, v, lat = data["test"]["known"], data["test"]["unknown"], data["verification"], data["embedding_latency_ms"]
        pct = lambda x: "n/a" if x is None else f"{x:.4f}"
        latency_text = "n/a" if lat["mean_ms"] is None else f"{lat['mean_ms']:.2f}/{lat['median_ms']:.2f}/{lat['p95_ms']:.2f}"
        lines.append(f"| {name} | {pct(k['conditional_rank1_accuracy'])} | {pct(k['end_to_end_rank1_accuracy'])} | "
                     f"{pct(k['conditional_correct_acceptance_rate'])} | {pct(k['end_to_end_correct_acceptance_rate'])} | "
                     f"{pct(u['conditional_rejection_rate'])} | {pct(u['conditional_fpir'])} | {pct(u['end_to_end_false_identification_rate'])} | "
                     f"{pct(k['conditional_fnir'])} | {pct(k['end_to_end_fnir'])} | {pct(v['eer'])} | {pct(v['roc_auc'])} | {latency_text} | "
                     f"{data['model_bytes'] / 1e6:.1f} | {sum(data['test']['detection_and_embedding_failures'].values())} |")
    lines += ["", "## Failure counts", ""]
    for name, data in report["models"].items():
        lines.append(f"- **{name}:** `{json.dumps(data['test']['detection_and_embedding_failures'], sort_keys=True)}`")
    lines += ["", "## Model licensing", "",
              "ArcFace antelopev2 weights are limited to non-commercial research absent separate authorization. AdaFace ONNX code is MIT, but its release does not grant a distinct weights license; WebFace4M model/data rights remain unresolved. Review both before deployment. SFace uses the existing OpenCV Zoo model; its package license does not independently settle upstream weight-training provenance.", ""]
    (out / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved complete reports and predictions: {out}", flush=True)


if __name__ == "__main__":
    main()
