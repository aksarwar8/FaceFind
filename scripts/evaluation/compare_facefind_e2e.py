"""Isolated FaceFind end-to-end LFW and synthetic group comparison.

System A is YuNet+SFace; System B is SCRFD-10G-KPS+AdaFace IR-18.
All outputs are created below data/private/evaluation and existing directories
are never overwritten. Run with the existing .venv_eval Python.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation import compare_widerface_detectors as detector_adapters
from evaluation.compare_open_set_recognizers import Encoder, sha256_file

OUT = ROOT / "data/private/evaluation/final_end_to_end_comparison_20261124_01"
TEST_MANIFEST = ROOT / "data/private/evaluation/open_set_run_20261123_v2/reproducibility_manifest.json"
CAL_SOURCE = ROOT / "data/private/evaluation/recognition_comparison_sface_arcface_adaface_full_20261123_01/calibration_split.json"
LFw_ROOT = ROOT / "data/benchmarks/lfw/lfw"
YU_MODEL = ROOT / "models/face/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = ROOT / "models/face/face_recognition_sface_2021dec.onnx"
SCRFD_MODEL = ROOT / "data/private/models/scrfd_10g_bnkps.onnx"
ADAFACE_MODEL = ROOT / "data/private/models/adaface_ir_18_webface4m_yakhyo.onnx"
SEED = 42
KNOWN_CAL_TARGET = 200
UNKNOWN_CAL_TARGET = 40
SUPPORTED = {".jpg", ".jpeg", ".png"}


def digest_text(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def face_area(face):
    x1, y1, x2, y2 = map(float, face["bbox"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_target(faces, width, height):
    """Largest face; ties by closest center, then top/left. Never uses identity."""
    if not faces:
        return None, None
    center = np.array([width / 2, height / 2])
    order = sorted(range(len(faces)), key=lambda i: (
        -face_area(faces[i]),
        float(np.linalg.norm((np.array(faces[i]["bbox"][:2]) + np.array(faces[i]["bbox"][2:])) / 2 - center)),
        float(faces[i]["bbox"][1]), float(faces[i]["bbox"][0])))
    return faces[order[0]], order[0]


def as_embed_face(face, system):
    x1, y1, x2, y2 = map(float, face["bbox"])
    # Existing SFace FaceRecognizerSF API takes YuNet xywh; SCRFD/AdaFace uses landmarks.
    bbox = [x1, y1, x2 - x1, y2 - y1] if system == "A" else [x1, y1, x2, y2]
    return {"bbox": bbox, "landmarks": face["landmarks"], "confidence": face["confidence"]}


class StridePaddedSCRFD:
    """Pad only the detector tensor to SCRFD's stride; coordinates stay in source pixels."""
    def __init__(self, adapter, stride=32):
        self.adapter, self.stride = adapter, stride

    def detect(self, image):
        height, width = image.shape[:2]
        padded_height = int(math.ceil(height / self.stride) * self.stride)
        padded_width = int(math.ceil(width / self.stride) * self.stride)
        if (padded_height, padded_width) == (height, width):
            return self.adapter.detect(image)
        padded = np.zeros((padded_height, padded_width, image.shape[2]), dtype=image.dtype)
        padded[:height, :width] = image
        return self.adapter.detect(padded)


class System:
    def __init__(self, key, detector, encoder, out_dir):
        self.key, self.detector, self.encoder = key, detector, encoder
        self.cache_dir = out_dir / "cache" / ("system_" + key.lower())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.detections = {}
        self.detection_errors = {}
        self.embeddings = {}
        self.detection_ms = {}
        self.embedding_ms = {}
        self.failures = []

    def detect(self, path, image=None):
        key = str(Path(path).resolve())
        if key in self.detections:
            return self.detections[key], self.detection_errors.get(key)
        if image is None:
            image = cv2.imread(key)
        if image is None:
            return [], {"failure": "image_read_failure", "image": key}
        start = time.perf_counter()
        try:
            faces = self.detector.detect(image)
            elapsed = (time.perf_counter() - start) * 1000
            self.detection_ms[key] = elapsed
            clean = []
            for row in faces:
                box = [float(x) for x in row["bbox"]]
                landmarks = [float(x) for x in row["landmarks"]]
                confidence = float(row["confidence"])
                if (len(box) != 4 or len(landmarks) != 10 or
                        not np.isfinite(box + landmarks + [confidence]).all()):
                    raise ValueError("invalid bbox, five-point landmarks or confidence")
                clean.append({"bbox": box, "landmarks": landmarks, "confidence": confidence})
            self.detections[key] = clean
            return clean, None
        except Exception as exc:
            self.detections[key] = []
            error = {"failure": "detection_failure", "image": key, "detail": str(exc)}
            self.detection_errors[key] = error
            return [], error

    def embed(self, path, face, face_index, image=None):
        key = str(Path(path).resolve()) + f"#{face_index}"
        if key in self.embeddings:
            return self.embeddings[key], None
        if image is None:
            image = cv2.imread(str(path))
        if image is None:
            return None, {"failure": "image_read_failure", "image": str(path)}
        try:
            started = time.perf_counter()
            vector, aligned = self.encoder.embed(image, as_embed_face(face, self.key))
            self.embedding_ms[key] = (time.perf_counter() - started) * 1000
            self.embeddings[key] = vector
            np.savez_compressed(self.cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".npz"),
                                embedding=vector, aligned_bgr=aligned)
            return vector, None
        except Exception as exc:
            return None, {"failure": "embedding_failure", "image": str(path), "detail": str(exc)}

    def detections_json(self):
        return {"detections": self.detections, "detector_errors": self.detection_errors}


def scores_for_gallery(gallery, vector):
    if not gallery:
        return None, -1.0, -1.0, -1.0
    names = list(gallery)
    matrix = np.vstack([gallery[n] for n in names])
    scores = matrix @ vector
    order = np.argsort(scores)[::-1]
    best, second = float(scores[order[0]]), float(scores[order[1]]) if len(order) > 1 else -1.0
    return names[int(order[0])], best, second, best - second


def predict(gallery, vector, threshold, margin):
    identity, score, second, gap = scores_for_gallery(gallery, vector)
    accepted = identity is not None and score >= threshold and gap >= margin
    return {"identity": identity if accepted else "unknown", "rank1_identity": identity,
            "best_score": score, "second_score": second, "margin": gap, "accepted": bool(accepted)}


def calibrate(gallery, known_records, unknown_records, system, threshold_grid=None, margin_grid=None):
    """Balanced correct-known acceptance/unknown rejection; conservative tie breaks."""
    known_samples, unknown_samples, failures = [], [], []
    for item in known_records:
        if item["identity"] not in gallery:
            continue
        for p in item["probes"]:
            image = cv2.imread(str(p))
            faces, err = system.detect(p, image)
            target, index = select_target(faces, image.shape[1], image.shape[0]) if image is not None else (None, None)
            if err or target is None:
                failures.append({"identity": item["identity"], "image": str(p), **(err or {"failure": "no_face_detected"})})
                continue
            emb, err = system.embed(p, target, index, image)
            if err:
                failures.append({"identity": item["identity"], "image": str(p), **err})
                continue
            top, score, second, gap = scores_for_gallery(gallery, emb)
            known_samples.append({"truth": item["identity"], "top": top, "score": score, "margin": gap,
                                  "ambiguous": len(faces) > 1})
    for item in unknown_records:
        for p in item["probes"]:
            image = cv2.imread(str(p))
            faces, err = system.detect(p, image)
            target, index = select_target(faces, image.shape[1], image.shape[0]) if image is not None else (None, None)
            if err or target is None:
                failures.append({"identity": item["identity"], "image": str(p), **(err or {"failure": "no_face_detected"})})
                continue
            emb, err = system.embed(p, target, index, image)
            if err:
                failures.append({"identity": item["identity"], "image": str(p), **err})
                continue
            top, score, second, gap = scores_for_gallery(gallery, emb)
            unknown_samples.append({"top": top, "score": score, "margin": gap,
                                    "ambiguous": len(faces) > 1})
    if not known_samples or not unknown_samples:
        raise RuntimeError(f"{system.key}: calibration lacks valid known/unknown scores; "
                           f"known={len(known_samples)} unknown={len(unknown_samples)}")
    if threshold_grid is None:
        values = np.asarray([x["score"] for x in known_samples + unknown_samples], dtype=float)
        threshold_grid = np.unique(np.quantile(values, np.linspace(0, 1, 51)))
    if margin_grid is None:
        values = np.asarray([x["margin"] for x in known_samples + unknown_samples], dtype=float)
        margin_grid = np.unique(np.quantile(values, np.linspace(0, 1, 31)))
    best = None
    for threshold in threshold_grid:
        for margin in margin_grid:
            known_correct = sum(x["top"] == x["truth"] and x["score"] >= threshold and x["margin"] >= margin
                                for x in known_samples) / len(known_samples)
            unknown_reject = sum(x["score"] < threshold or x["margin"] < margin
                                 for x in unknown_samples) / len(unknown_samples)
            objective = (known_correct + unknown_reject) / 2
            candidate = (objective, float(threshold), float(margin), known_correct, unknown_reject)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    _, threshold, margin, known_correct, unknown_reject = best
    return {"threshold": threshold, "margin": margin, "known_correct_accept_rate": known_correct,
            "unknown_rejection_rate": unknown_reject, "objective": best[0],
            "known_samples": len(known_samples), "unknown_samples": len(unknown_samples),
            "calibration_failures": failures,
            "ambiguous_known": sum(x["ambiguous"] for x in known_samples),
            "ambiguous_unknown": sum(x["ambiguous"] for x in unknown_samples)}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0


def greedy_match(gt, preds, threshold=0.5):
    matched, pairs = set(), []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["confidence"]):
        best, overlap = None, 0.0
        for gi, g in enumerate(gt):
            if gi in matched:
                continue
            value = iou(preds[pi]["bbox"], g["bbox"])
            if value > overlap:
                best, overlap = gi, value
        if best is not None and overlap >= threshold:
            matched.add(best)
            pairs.append((best, pi, overlap))
    return pairs, set(range(len(gt))) - matched, set(range(len(preds))) - {p for _, p, _ in pairs}


def haar_box(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(boxes) == 0:
        return None, 0
    boxes = [list(map(int, b)) for b in boxes]
    h, w = gray.shape
    boxes.sort(key=lambda b: (-(b[2] * b[3]),
                              math.hypot(b[0] + b[2]/2 - w/2, b[1] + b[3]/2 - h/2), b[1], b[0]))
    return boxes[0], len(boxes)


def choose_donor(identity, paths, gallery_path, cascade, donor_cache):
    for path in sorted(map(Path, paths)):
        if str(path.resolve()) == str(Path(gallery_path).resolve()):
            continue
        key = str(path.resolve())
        image = cv2.imread(key)
        if image is None:
            continue
        box, count = haar_box(image, cascade)
        if box is None:
            continue
        x, y, w, h = box
        margin_x, margin_y = max(4, int(w * .30)), max(4, int(h * .30))
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2, y2 = min(image.shape[1], x + w + margin_x), min(image.shape[0], y + h + margin_y)
        crop = image[y1:y2, x1:x2].copy()
        if crop.size:
            donor_cache[identity] = {"identity": identity, "source_image": key, "gallery_image": str(gallery_path),
                                     "crop": crop, "face_box_in_crop": [x-x1, y-y1, x+w-x1, y+h-y1],
                                     "haar_source_faces": count}
            return donor_cache[identity]
    return None


def build_group_photos(test_manifest, cal_known, out_dir, seed=SEED):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        raise FileNotFoundError(f"OpenCV Haar cascade missing: {cascade_path}")
    cascade = cv2.CascadeClassifier(str(cascade_path))
    rng = random.Random(seed)
    donor_cache = {}
    known_items = list(test_manifest["known"].items())
    unknown_items = list(test_manifest["unknown"].items())
    rng.shuffle(known_items)
    rng.shuffle(unknown_items)
    known_donors, unknown_donors = [], []
    for identity, item in known_items:
        donor = choose_donor(identity, item["probe_images"], item["gallery_image"], cascade, donor_cache)
        if donor:
            known_donors.append(donor)
        if len(known_donors) == 50:
            break
    for identity, paths in unknown_items:
        donor = choose_donor(identity, paths, "", cascade, donor_cache)
        if donor:
            unknown_donors.append(donor)
        if len(unknown_donors) == 50:
            break
    if len(known_donors) < 50 or len(unknown_donors) < 50:
        raise RuntimeError(f"Need 50 known/unknown donors; got {len(known_donors)}/{len(unknown_donors)}")
    all_donors = []
    for gi in range(10):
        all_donors.append(known_donors[gi*5:(gi+1)*5] + unknown_donors[gi*5:(gi+1)*5])
    group_dir = out_dir / "group_photos"
    examples_dir = out_dir / "annotated_group_examples"
    group_dir.mkdir()
    examples_dir.mkdir()
    ann = []
    scales = [1.0, .8, .65, .5, .4, .32, .25, .2, .17, .14]
    resolutions = [1.0, 1.0, .8, .8, .65, .65, .5, .8, 1.0, .5]
    lighting = [.95, 1.1, .72, 1.0, .85, 1.15, .68, 1.0, .9, 1.08]
    for group_index, people in enumerate(all_donors):
        canvas = np.zeros((640, 1600, 3), np.uint8)
        bg = [int(130 + (group_index*19)%70), int(145 + (group_index*13)%55), int(155 + (group_index*11)%45)]
        canvas[:] = bg
        gt_faces = []
        for slot, donor in enumerate(people):
            row, col = divmod(slot, 5)
            xoff, yoff = col * 320, row * 320
            chip = donor["crop"]
            b = donor["face_box_in_crop"]
            target_h = [64, 52, 42, 32, 24][(slot + group_index) % 5]
            factor = target_h / max(1, b[3] - b[1])
            new_w, new_h = max(1, int(round(chip.shape[1] * factor))), max(1, int(round(chip.shape[0] * factor)))
            if new_w > 290 or new_h > 290:
                shrink = min(290 / new_w, 290 / new_h)
                factor *= shrink
                new_w, new_h = int(round(chip.shape[1] * factor)), int(round(chip.shape[0] * factor))
            resized = cv2.resize(chip, (new_w, new_h), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
            px, py = xoff + (320-new_w)//2, yoff + (320-new_h)//2
            canvas[py:py+new_h, px:px+new_w] = resized
            gt = [px + b[0]*factor, py + b[1]*factor, px + b[2]*factor, py + b[3]*factor]
            gt_faces.append({"identity": donor["identity"], "known": slot < 5, "bbox": [round(v, 3) for v in gt],
                             "source_image": donor["source_image"], "source_gallery_image": donor["gallery_image"],
                             "source_haar_face_count": donor["haar_source_faces"],
                             "slot": slot, "target_face_height_px_before_scene_resize": round(gt[3]-gt[1], 2)})
        if resolutions[group_index] != 1.0:
            scale = resolutions[group_index]
            canvas = cv2.resize(canvas, (int(1600*scale), int(640*scale)), interpolation=cv2.INTER_AREA)
            for gt in gt_faces:
                gt["bbox"] = [round(v*scale, 3) for v in gt["bbox"]]
                gt["target_face_height_px"] = round((gt["bbox"][3]-gt["bbox"][1]), 2)
        else:
            for gt in gt_faces:
                gt["target_face_height_px"] = gt["target_face_height_px_before_scene_resize"]
        canvas = np.clip(canvas.astype(np.float32) * lighting[group_index], 0, 255).astype(np.uint8)
        if group_index in (2, 5, 8):
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0.55)
        filename = f"group_{group_index+1:02d}.jpg"
        cv2.imwrite(str(group_dir / filename), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ann.append({"image": f"group_photos/{filename}", "width": canvas.shape[1], "height": canvas.shape[0],
                    "faces": gt_faces, "synthetic": True, "lighting_factor": lighting[group_index],
                    "blurred": group_index in (2, 5, 8)})
    return ann, {"cascade": str(cascade_path), "cascade_sha256": sha256_file(cascade_path),
                 "known_identities": [d["identity"] for d in known_donors],
                 "unknown_identities": [d["identity"] for d in unknown_donors],
                 "donor_manifest": [{k:v for k,v in d.items() if k not in {"crop", "face_box_in_crop"}}
                                    for d in known_donors+unknown_donors]}


def calibration_build(test, calibration, systems, out_dir):
    excluded = set(test["known"]) | set(test["unknown"])
    used = set()
    cal_known = []
    rejected_refs = []

    def shared_reference_ok(path):
        image = cv2.imread(str(path))
        if image is None:
            return False, {"failure": "image_read_failure"}
        for obj in systems.values():
            faces, err = obj.detect(path, image)
            if err or not faces:
                return False, err or {"failure": "no_face_detected", "system": obj.key}
            face, idx = select_target(faces, image.shape[1], image.shape[0])
            _, emb_err = obj.embed(path, face, idx, image)
            if emb_err:
                return False, {**emb_err, "system": obj.key}
        return True, None
    for item_index, item in enumerate(calibration["known"], 1):
        identity = item["identity"]
        if identity in excluded or identity in used:
            continue
        candidate_paths = [Path(item["gallery"]), *map(Path, item["probes"])]
        selected = None
        for ref in candidate_paths:
            ok, reason = shared_reference_ok(ref)
            if ok:
                selected = ref
                break
            rejected_refs.append({"identity": identity, "image": str(ref), **(reason or {})})
        if selected is None:
            continue
        probes = [p for p in candidate_paths if p.resolve() != selected.resolve()]
        cal_known.append({"identity": identity, "gallery": selected, "probes": probes})
        used.add(identity)
        if item_index % 25 == 0 or item_index == len(calibration["known"]):
            print(f"Calibration gallery candidates checked {item_index}/{len(calibration['known'])}; usable {len(cal_known)}",flush=True)
        elif item_index % 5 == 0:
            print(f"Calibration gallery candidates checked {item_index}/{len(calibration['known'])}; usable {len(cal_known)}",flush=True)
    # Reserve identities outside test and original calibration cohorts only when needed.
    all_existing = set(x["identity"] for x in calibration["known"] + calibration["unknown"])
    if len(cal_known) < KNOWN_CAL_TARGET:
        reserves = []
        for folder in sorted(LFw_ROOT.iterdir()):
            if not folder.is_dir() or folder.name in excluded | all_existing | used:
                continue
            paths = sorted(p.resolve() for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
            if len(paths) >= 2:
                reserves.append((folder.name, paths))
        for identity, paths in reserves:
            selected = None
            for ref in paths:
                ok, reason = shared_reference_ok(ref)
                if ok:
                    selected = ref
                    break
                rejected_refs.append({"identity": identity, "image": str(ref), **(reason or {})})
            if selected:
                cal_known.append({"identity": identity, "gallery": selected,
                                  "probes": [p for p in paths if p != selected]})
                used.add(identity)
                if len(cal_known) >= KNOWN_CAL_TARGET:
                    break
    if len(cal_known) < KNOWN_CAL_TARGET:
        raise RuntimeError(f"Only {len(cal_known)} shared usable calibration gallery identities; target {KNOWN_CAL_TARGET}")
    cal_known = cal_known[:KNOWN_CAL_TARGET]
    cal_unknown = [{"identity": x["identity"], "probes": [Path(p) for p in x["probes"]]}
                   for x in calibration["unknown"] if x["identity"] not in excluded and x["identity"] not in used]
    # Keep the existing unknown calibration identities and replace only identities with no usable probe faces.
    accepted_unknown = []
    for item in cal_unknown:
        usable = any(all(not s.detect(p)[1] and s.detect(p)[0] for s in systems.values()) for p in item["probes"])
        if usable:
            accepted_unknown.append(item)
    if len(accepted_unknown) < UNKNOWN_CAL_TARGET:
        excluded_reserve = excluded | used | {x["identity"] for x in accepted_unknown} | all_existing
        for folder in sorted(LFw_ROOT.iterdir()):
            if len(accepted_unknown) >= UNKNOWN_CAL_TARGET:
                break
            if not folder.is_dir() or folder.name in excluded_reserve:
                continue
            paths = sorted(p.resolve() for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
            if len(paths) >= 2 and any(all(not s.detect(p)[1] and s.detect(p)[0] for s in systems.values()) for p in paths):
                accepted_unknown.append({"identity": folder.name, "probes": paths})
                excluded_reserve.add(folder.name)
    if len(accepted_unknown) < UNKNOWN_CAL_TARGET:
        raise RuntimeError(f"Only {len(accepted_unknown)} usable unknown calibration identities; target {UNKNOWN_CAL_TARGET}")
    return cal_known, accepted_unknown[:UNKNOWN_CAL_TARGET], rejected_refs


def make_gallery(records, system, stage):
    gallery, failures, ambiguities = {}, [], []
    for record in records:
        path = Path(record["gallery"] if "gallery" in record else record["gallery_image"])
        image = cv2.imread(str(path))
        faces, err = system.detect(path, image)
        target, idx = select_target(faces, image.shape[1], image.shape[0]) if image is not None else (None, None)
        if err or target is None:
            failures.append({"identity": record["identity"], "image": str(path), **(err or {"failure": "no_face_detected"})})
            continue
        if len(faces) > 1:
            ambiguities.append({"identity": record["identity"], "image": str(path), "face_count": len(faces), "selected_index": idx})
        vector, e = system.embed(path, target, idx, image)
        if e:
            failures.append({"identity": record["identity"], "image": str(path), **e})
        else:
            gallery[record["identity"]] = vector
    return gallery, failures, ambiguities


def evaluate_lfw(test, systems, thresholds, out_dir):
    results = {}
    probes = []
    for identity, row in test["known"].items():
        probes.extend([{"split":"known","truth":identity,"image":str(p)} for p in row["probe_images"]])
    for identity, paths in test["unknown"].items():
        probes.extend([{"split":"unknown","truth":identity,"image":str(p)} for p in paths])
    for sysobj in systems.values():
        key = sysobj.key
        gallery_records = [{"identity":i,"gallery_image":r["gallery_image"]} for i,r in test["known"].items()]
        gallery, gallery_failures, gallery_ambiguous = make_gallery(gallery_records, sysobj, "test_gallery")
        rows, amb, failures = [], [], []
        prediction_file = out_dir / f"lfw_predictions_system_{key.lower()}.jsonl"
        with prediction_file.open("x", encoding="utf-8") as f:
            for n, probe in enumerate(probes,1):
                path = Path(probe["image"]); image = cv2.imread(str(path))
                faces, err = sysobj.detect(path,image)
                target,idx = select_target(faces,image.shape[1],image.shape[0]) if image is not None else (None,None)
                if len(faces)>1:
                    amb.append({"image":str(path),"split":probe["split"],"truth":probe["truth"],"face_count":len(faces),"selected_index":idx})
                if err or target is None:
                    reason=err or {"failure":"no_face_detected"}
                    row={**probe,"status":"failed",**reason}
                    failures.append(row)
                else:
                    emb,e=sysobj.embed(path,target,idx,image)
                    if e:
                        row={**probe,"status":"failed",**e}; failures.append(row)
                    else:
                        pred=predict(gallery,emb,thresholds[key]["lfw"]["threshold"],thresholds[key]["lfw"]["margin"])
                        row={**probe,"status":"evaluated","selected_face_index":idx,"face_count":len(faces),**pred}
                rows.append(row); f.write(json.dumps(row)+"\n")
                if n%100==0 or n==len(probes): print(f"[System {key}] LFW probes {n}/{len(probes)}",flush=True)
        known=[r for r in rows if r["split"]=="known"]; unknown=[r for r in rows if r["split"]=="unknown"]
        ke=[r for r in known if r["status"]=="evaluated"]; ue=[r for r in unknown if r["status"]=="evaluated"]
        correct_rank=sum(r["rank1_identity"]==r["truth"] for r in ke)
        correct_accept=sum(r["accepted"] and r["rank1_identity"]==r["truth"] for r in ke)
        false_accept=sum(r["accepted"] for r in ue)
        results[key]={"gallery_size_expected":200,"gallery_size_usable":len(gallery),"gallery_failures":gallery_failures,
                      "gallery_ambiguous":gallery_ambiguous,"probe_count":len(rows),"known_probes":len(known),
                      "unknown_probes":len(unknown),"evaluated_known":len(ke),"evaluated_unknown":len(ue),
                      "rank1_conditional":correct_rank/len(ke) if ke else None,
                      "rank1_end_to_end":correct_rank/len(known),
                      "known_correct_accept_conditional":correct_accept/len(ke) if ke else None,
                      "known_correct_accept_end_to_end":correct_accept/len(known),
                      "unknown_reject_conditional":(len(ue)-false_accept)/len(ue) if ue else None,
                      "unknown_reject_end_to_end":((len(ue)-false_accept)/len(unknown)),
                      "fpir_conditional":false_accept/len(ue) if ue else None,"fpir_end_to_end":false_accept/len(unknown),
                      "fnir_conditional":1-correct_accept/len(ke) if ke else None,"fnir_end_to_end":1-correct_accept/len(known),
                      "ambiguous_probes":amb,"ambiguous_probe_count":len(amb),"failures":failures,
                      "failure_counts":dict(Counter(r["failure"] for r in failures)),
                      "detection_latency":lat_summary([sysobj.detection_ms[x["image"]] for x in probes if x["image"] in sysobj.detection_ms]),
                      "recognition_latency":lat_summary([v for k,v in sysobj.embedding_ms.items() if any(k.startswith(x["image"]+"#") for x in probes)]),
                      "prediction_file":prediction_file.name}
    return results, probes


def lat_summary(values):
    if not values:return {"count":0,"mean_ms":None,"median_ms":None,"p95_ms":None}
    return {"count":len(values),"mean_ms":float(np.mean(values)),"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95))}


def evaluate_groups(group_annotations, systems, group_gallery, thresholds, out_dir):
    summaries={}
    for sysobj in systems.values():
        key=sysobj.key; totals=Counter(); idstats=Counter(); duplicates=0; failed=[]; examples=[]
        group_det_samples=[]; group_embed_samples=[]
        pfile=out_dir/f"group_predictions_system_{key.lower()}.jsonl"
        with pfile.open("x",encoding="utf-8") as stream:
            for scene in group_annotations:
                path=out_dir/scene["image"]; image=cv2.imread(str(path)); start=time.perf_counter()
                faces,err=sysobj.detect(path,image)
                if str(path.resolve()) in sysobj.detection_ms:
                    group_det_samples.append(sysobj.detection_ms[str(path.resolve())])
                if err:
                    failed.append({"image":scene["image"],**err}); faces=[]
                vectors=[]; face_errors=[]
                for i,face in enumerate(faces):
                    emb,e=sysobj.embed(path,face,i,image)
                    emb_key=str(path.resolve())+f"#{i}"
                    if emb_key in sysobj.embedding_ms:
                        group_embed_samples.append(sysobj.embedding_ms[emb_key])
                    vectors.append(emb); face_errors.append(e)
                pairs,missed,extra=greedy_match(scene["faces"],faces)
                totals["gt_faces"]+=len(scene["faces"]); totals["detections"]+=len(faces)
                totals["tp"]+=len(pairs); totals["fn"]+=len(missed); totals["fp"]+=len(extra)
                id_by_gt={}
                face_pred={}
                for gi,pi,overlap in pairs:
                    truth=scene["faces"][gi]["identity"]; is_known=scene["faces"][gi]["known"]
                    if is_known:idstats["known_gt_detected"]+=1
                    else:idstats["unknown_gt_detected"]+=1
                    if face_errors[pi]:
                        idstats["embedding_failures"]+=1
                        face_pred[pi]={"identity":"error","failure":face_errors[pi]}
                        failed.append({"image":scene["image"],"face_index":pi,**face_errors[pi]})
                        continue
                    pred=predict(group_gallery[key],vectors[pi],thresholds[key]["group"]["threshold"],thresholds[key]["group"]["margin"])
                    face_pred[pi]=pred; id_by_gt[gi]=pred
                    if is_known:
                        if pred["rank1_identity"]==truth:idstats["known_rank1_correct"]+=1
                        if pred["accepted"] and pred["rank1_identity"]==truth:idstats["known_correct_accept"]+=1
                        elif pred["accepted"]: idstats["incorrect_identity_labels"]+=1
                        else:idstats["known_rejected"]+=1
                    else:
                        if pred["accepted"]:idstats["unknown_false_accept"]+=1
                        else:idstats["unknown_rejected"]+=1
                idstats["known_gt_total"]+=sum(x["known"] for x in scene["faces"])
                idstats["unknown_gt_total"]+=sum(not x["known"] for x in scene["faces"])
                # Different ground-truth people assigned the same accepted known identity in this photo.
                assigned=defaultdict(set)
                for gi,pred in id_by_gt.items():
                    if pred["accepted"]:assigned[pred["identity"]].add(scene["faces"][gi]["identity"])
                duplicates+=sum(max(0,len(ids)-1) for ids in assigned.values())
                row={"image":scene["image"],"status":"evaluated","ground_truth_faces":len(scene["faces"]),
                     "detections":len(faces),"matched":len(pairs),"missed_gt_indices":sorted(missed),"fp_detection_indices":sorted(extra),
                     "matches":[{"gt_index":g,"pred_index":p,"iou":ov,"truth":scene["faces"][g]["identity"],**face_pred.get(p,{})} for g,p,ov in pairs],
                     "processing_ms":(time.perf_counter()-start)*1000}
                stream.write(json.dumps(row)+"\n")
                # Annotated private example image per architecture/scene.
                draw=image.copy()
                for g,p,ov in pairs:
                    b=faces[p]["bbox"]; x1,y1,x2,y2=map(int,b); lab=face_pred.get(p,{}).get("identity","embed-fail")
                    cv2.rectangle(draw,(x1,y1),(x2,y2),(30,220,40),2); cv2.putText(draw,str(lab),(x1,max(14,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,30,240),1)
                for gi in missed:
                    x1,y1,x2,y2=map(int,scene["faces"][gi]["bbox"]); cv2.rectangle(draw,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.imwrite(str(out_dir/"annotated_group_examples"/f"{key.lower()}_{Path(scene['image']).stem}.jpg"),draw)
                examples.append({"image":scene["image"],"matches":len(pairs),"misses":len(missed),
                                 "false_positives":len(extra),"processing_ms":row["processing_ms"]})
                print(f"[System {key}] group {Path(scene['image']).name}: {len(pairs)}/10 faces matched",flush=True)
        tp,fp,fn=totals["tp"],totals["fp"],totals["fn"]
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        summaries[key]={"images":len(group_annotations),"gt_faces":totals["gt_faces"],"detections":totals["detections"],
                        "tp":tp,"fp":fp,"fn":fn,"detection_precision":prec,"detection_recall":rec,
                        "detection_f1":2*prec*rec/(prec+rec) if prec+rec else 0,
                        "missed_faces":fn,"incorrect_identity_labels":idstats["incorrect_identity_labels"],
                        "embedding_failures":idstats["embedding_failures"],
                        "duplicate_identity_assignments":duplicates,
                        "known_people": {"total":idstats["known_gt_total"],"detected":idstats["known_gt_detected"],
                                         "correct_rank1":idstats["known_rank1_correct"],"correct_accepted":idstats["known_correct_accept"],
                                         "conditional_correct_identification_rate":idstats["known_correct_accept"]/idstats["known_gt_detected"] if idstats["known_gt_detected"] else None,
                                         "end_to_end_identification_rate":idstats["known_correct_accept"]/idstats["known_gt_total"] if idstats["known_gt_total"] else None},
                        "unknown_people":{"total":idstats["unknown_gt_total"],"detected":idstats["unknown_gt_detected"],
                                          "genuine_rejections":idstats["unknown_rejected"],"false_identifications":idstats["unknown_false_accept"],
                                          "conditional_rejection_rate":idstats["unknown_rejected"]/idstats["unknown_gt_detected"] if idstats["unknown_gt_detected"] else None,
                                          "end_to_end_rejection_rate":(idstats["unknown_rejected"]+fn*0)/idstats["unknown_gt_total"] if idstats["unknown_gt_total"] else None,
                                          "conditional_fpir":idstats["unknown_false_accept"]/idstats["unknown_gt_detected"] if idstats["unknown_gt_detected"] else None,
                                          "end_to_end_fpir":idstats["unknown_false_accept"]/idstats["unknown_gt_total"] if idstats["unknown_gt_total"] else None},
                        "detection_latency":lat_summary(group_det_samples),
                        "recognition_latency":lat_summary(group_embed_samples),
                        "group_processing_latency":lat_summary([x["processing_ms"] for x in examples]),
                        "failures":failed,"prediction_file":pfile.name,"annotated_examples":len(examples)}
    return summaries


def main():
    if OUT.exists():
        allowed = {"cache_attempt01_partial", "attempt01_partial_outputs"}
        existing = {p.name for p in OUT.iterdir()}
        if existing - allowed or not allowed.issubset(existing):
            raise FileExistsError(f"Refusing to overwrite existing output in {OUT}: {sorted(existing)}")
    for p in (TEST_MANIFEST,CAL_SOURCE,LFw_ROOT,YU_MODEL,SFACE_MODEL,SCRFD_MODEL,ADAFACE_MODEL):
        if not p.exists():raise FileNotFoundError(str(p))
    out=OUT
    out.mkdir(parents=True, exist_ok=True)
    (out/"cache").mkdir()
    test=json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    calibration=json.loads(CAL_SOURCE.read_text(encoding="utf-8"))
    known_probe_count=sum(len(x["probe_images"]) for x in test["known"].values())
    unknown_probe_count=sum(len(x) for x in test["unknown"].values())
    if len(test["known"])!=200 or len(test["unknown"])!=100 or known_probe_count!=756 or unknown_probe_count!=557:
        raise ValueError(f"Unexpected fixed LFW protocol: identities={len(test['known'])}/{len(test['unknown'])}, probes={known_probe_count}/{unknown_probe_count}")
    test_ids=set(test["known"])|set(test["unknown"])
    detector_a=detector_adapters.YuNetAdapter(score_threshold=.5,nms_threshold=.3,max_dimension=1280)
    detector_b=StridePaddedSCRFD(detector_adapters.create_scrfd_adapter(SCRFD_MODEL,score_threshold=.5,nms_threshold=.4,max_dimension=1280))
    system_a=System("A",detector_a,Encoder("SFace",SFACE_MODEL),out)
    system_b=System("B",detector_b,Encoder("AdaFace",ADAFACE_MODEL),out)
    systems={"A":system_a,"B":system_b}
    try:
        import psutil
        proc=psutil.Process(os.getpid())
        mem_start=proc.memory_info().rss
        mem_peak=mem_start
        system_mem_start=psutil.virtual_memory().available
        psutil_available=True
    except ImportError:
        psutil=None; proc=None; mem_start=None; mem_peak=None; system_mem_start=None; psutil_available=False

    def sample_memory():
        nonlocal mem_peak
        if proc is not None:
            mem_peak=max(mem_peak,proc.memory_info().rss)
    print("Both system adapters initialized; selecting shared disjoint calibration references.",flush=True)
    cal_known,cal_unknown,cal_ref_rejections=calibration_build(test,calibration,systems,out)
    assert len(cal_known)==200 and len(cal_unknown)==40
    assert not ({x["identity"] for x in cal_known+cal_unknown}&test_ids)
    cal_manifest={"known":[{"identity":x["identity"],"gallery":str(x["gallery"]),"probes":list(map(str,x["probes"]))} for x in cal_known],
                  "unknown":[{"identity":x["identity"],"probes":list(map(str,x["probes"]))} for x in cal_unknown],
                  "test_identity_overlap":[],"usable_known_gallery_count":len(cal_known),"rejected_gallery_candidates":cal_ref_rejections,"seed":SEED}
    (out/"calibration_results.json").write_text(json.dumps({"selected_split":cal_manifest},indent=2),encoding="utf-8")
    calibration_galleries={}; thresholds={}
    # Build and calibrate 200-way LFW and 50-way group-photo matchers per system.
    for key,s in systems.items():
        grecs=[{"identity":x["identity"],"gallery":x["gallery"]} for x in cal_known]
        gallery,gfail,gamb=make_gallery(grecs,s,"calibration")
        if len(gallery)!=200:raise RuntimeError(f"System {key} calibration embedding failures: {200-len(gallery)}; examples={gfail[:4]}")
        calibration_galleries[key]=gallery
        lfw=calibrate(gallery,cal_known,cal_unknown,s)
        group_gallery={name:gallery[name] for name in list(gallery)[:50]}
        group_cal_known=[x for x in cal_known if x["identity"] in group_gallery]
        group=calibrate(group_gallery,group_cal_known,cal_unknown,s)
        thresholds[key]={"lfw":lfw,"group":group}
        print(f"[System {key}] thresholds frozen: LFW={lfw['threshold']:.5f}/{lfw['margin']:.5f}; group={group['threshold']:.5f}/{group['margin']:.5f}",flush=True)
    frozen={k:{task:{"threshold":v["threshold"],"margin":v["margin"],"known_samples":v["known_samples"],"unknown_samples":v["unknown_samples"],
                     "ambiguous_known":v["ambiguous_known"],"ambiguous_unknown":v["ambiguous_unknown"],
                     "failure_counts":dict(Counter(x["failure"] for x in v["calibration_failures"]))}
              for task,v in t.items()} for k,t in thresholds.items()}
    (out/"frozen_thresholds.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    (out/"calibration_results.json").write_text(json.dumps({"selected_split":cal_manifest,
        "systems":{k:{task:{**v,"calibration_failures_count":len(v["calibration_failures"])} for task,v in t.items()}
                   for k,t in thresholds.items()}},indent=2),encoding="utf-8")
    print("All thresholds frozen before test scoring.",flush=True)
    lfw_results,probes=evaluate_lfw(test,systems,thresholds,out)
    sample_memory()
    group_annotations,group_source=build_group_photos(test,cal_known,out)
    (out/"group_annotations.json").write_text(json.dumps(group_annotations,indent=2),encoding="utf-8")
    (out/"group_source_manifest.json").write_text(json.dumps(group_source,indent=2),encoding="utf-8")
    group_reference_records=[{"identity":i,"gallery_image":test["known"][i]["gallery_image"]}
                             for i in group_source["known_identities"]]
    group_gallery={}; group_reference_failures={}
    for key,s in systems.items():
        group_gallery[key],group_reference_failures[key],_=make_gallery(group_reference_records,s,"group_gallery")
        if len(group_gallery[key])!=50:
            raise RuntimeError(f"System {key}: expected 50 usable group gallery references; got {len(group_gallery[key])}; failures={group_reference_failures[key][:3]}")
    group_results=evaluate_groups(group_annotations,systems,group_gallery,thresholds,out)
    sample_memory()
    for key in group_results:
        group_results[key]["gallery_size_expected"]=50
        group_results[key]["gallery_size_usable"]=len(group_gallery[key])
        group_results[key]["gallery_failures"]=group_reference_failures[key]
    for key,s in systems.items():
        (out/f"detections_system_{key.lower()}.json").write_text(json.dumps(s.detections_json()),encoding="utf-8")
    all_test_paths=[str(Path(x["image"]).resolve()) for x in probes]
    repro={"seed":SEED,"test_manifest":str(TEST_MANIFEST),"test_known_identities":len(test["known"]),"test_unknown_identities":len(test["unknown"]),
           "test_known_probes":sum(len(x["probe_images"]) for x in test["known"].values()),"test_unknown_probes":sum(len(x) for x in test["unknown"].values()),
           "test_probe_path_sha256":digest_text(all_test_paths),"calibration_file":str(out/"calibration_results.json"),"calibration_identity_overlap":[],
           "calibration_usable_gallery_target":200,"group_protocol":"synthetic; 10 mosaics x 10 people (5 known, 5 unknown); Haar-derived source boxes; held-out probe crops only",
           "group_known_reference_count":50,"group_unknown_identity_count":50,"group_source_gallery_images_reused_as_test":False,
           "target_face_selection":"largest bbox; center-distance tie break; top-left final tie break; no identity/score use",
           "landmarks_and_embedding":{"system_a":"YuNet native five-point order passed to OpenCV FaceRecognizerSF.alignCrop; SFace feature L2-normalized",
                                       "system_b":"SCRFD KPS order (left eye, right eye, nose, left mouth, right mouth); 112x112 ArcFace template, BGR NCHW float32, (pixel-127.5)/127.5; AdaFace embedding L2-normalized",
                                       "matching":"cosine similarity; accept only when top score >= frozen threshold and top1-top2 >= frozen margin"},
           "runtime":{"python":sys.version,"opencv":cv2.__version__,"onnxruntime":ort.__version__,"numpy":np.__version__,"cpu_execution":True},
           "systems":{"A":{"detector":"YuNet","detector_score_threshold":.5,"detector_nms_threshold":.3,"recognizer":"SFace"},
                      "B":{"detector":"SCRFD-10G-KPS","detector_score_threshold":.5,"detector_nms_threshold":.4,"recognizer":"AdaFace IR-18"}}}
    (out/"reproducibility_manifest.json").write_text(json.dumps(repro,indent=2),encoding="utf-8")
    result={"status":"complete","test_manifest":str(TEST_MANIFEST),"test_probe_count":len(probes),"test_probe_path_sha256":repro["test_probe_path_sha256"],
            "resource_usage":{"psutil_available":psutil_available,"process_rss_start_bytes":mem_start,
                              "process_rss_peak_sampled_bytes":mem_peak,
                              "system_available_memory_start_bytes":system_mem_start,
                              "system_available_memory_end_bytes":psutil.virtual_memory().available if psutil_available else None},
            "calibration":{"known_gallery_identities":200,"unknown_identities":40,"identity_overlap":[],"thresholds":frozen},
            "systems":{"A":{"name":"YuNet + SFace","lfw":lfw_results["A"],"synthetic_groups":group_results["A"],"models":{"detector_bytes":YU_MODEL.stat().st_size,"recognizer_bytes":SFACE_MODEL.stat().st_size,"detector_sha256":sha256_file(YU_MODEL),"recognizer_sha256":sha256_file(SFACE_MODEL)}},
                       "B":{"name":"SCRFD-10G-KPS + AdaFace IR-18","lfw":lfw_results["B"],"synthetic_groups":group_results["B"],"models":{"detector_bytes":SCRFD_MODEL.stat().st_size,"recognizer_bytes":ADAFACE_MODEL.stat().st_size,"detector_sha256":sha256_file(SCRFD_MODEL),"recognizer_sha256":sha256_file(ADAFACE_MODEL)}}},
            "group_evaluation":{"synthetic":True,"images":len(group_annotations),"ground_truth_faces":sum(len(x["faces"]) for x in group_annotations),
                                "not_representative_of_real_world_group_accuracy":True},
            "licensing":{"scrfd":"non-commercial research restriction; separate permission needed for commercial use",
                         "adaface":"MIT repo code; model checkpoint license/training data rights unresolved"}}
    a=result["systems"]["A"]; b=result["systems"]["B"]
    a_score=(a["lfw"].get("known_correct_accept_end_to_end") or 0)+(a["synthetic_groups"]["known_people"].get("end_to_end_identification_rate") or 0)
    b_score=(b["lfw"].get("known_correct_accept_end_to_end") or 0)+(b["synthetic_groups"]["known_people"].get("end_to_end_identification_rate") or 0)
    if b_score>a_score and b["synthetic_groups"]["group_processing_latency"]["mean_ms"] < a["synthetic_groups"]["group_processing_latency"]["mean_ms"]:
        result["recommendation"]="System B leads on combined measured known-person end-to-end accuracy and group-photo processing speed, but SCRFD is non-commercial research-only and AdaFace checkpoint rights are unresolved; use System B only for research pending licensing clearance."
    elif a_score>=b_score:
        result["recommendation"]="System A is the measured practical choice on combined known-person accuracy; it also avoids System B's explicit SCRFD non-commercial restriction and AdaFace checkpoint licensing uncertainty. Confirm SFace upstream weight provenance before deployment."
    else:
        result["recommendation"]="System A remains the deployment candidate pending SFace weight provenance review; System B's accuracy/speed trade-off is research-only until SCRFD and AdaFace rights are cleared."
    (out/"comparison_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    write_markdown(out,result)
    print(f"Complete evaluation saved: {out}",flush=True)


def write_markdown(out,result):
    names={"A":"YuNet + SFace","B":"SCRFD-10G-KPS + AdaFace IR-18"}
    lines=["# FaceFind end-to-end architecture comparison","",
           "LFW uses the fixed existing test manifest. Synthetic group-photo metrics are reported separately and do not represent real-world group-photo accuracy.","",
           "## LFW open-set identification","",
           "| System | Rank-1 conditional | Rank-1 end-to-end | Known correct+accept conditional | Known correct+accept end-to-end | Unknown rejection conditional | Unknown rejection end-to-end | FPIR | FNIR | Gallery failures | Ambiguous probes |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("A","B"):
        d=result["systems"][k]["lfw"]; pct=lambda x:"n/a" if x is None else f"{x:.4f}"
        lines.append(f"| {names[k]} | {pct(d['rank1_conditional'])} | {pct(d['rank1_end_to_end'])} | {pct(d['known_correct_accept_conditional'])} | {pct(d['known_correct_accept_end_to_end'])} | {pct(d['unknown_reject_conditional'])} | {pct(d['unknown_reject_end_to_end'])} | {pct(d['fpir_conditional'])} | {pct(d['fnir_conditional'])} | {len(d['gallery_failures'])} | {d['ambiguous_probe_count']} |")
    lines += ["","## Synthetic group-photo results","",
              "Ten controlled composites contain 100 held-out people total: 50 known identities and 50 unknown identities, ten per image. Crops came from probe images, never enrollment images. Haar boxes were transformed into composite coordinates as synthetic annotations. Lighting, blur, face size and output resolution vary. These results measure the controlled composites only.","",
              "| System | Detection P / R / F1 | Known correct ID conditional / end-to-end | Unknown rejection conditional / end-to-end | FPIR conditional / end-to-end | FNIR conditional / end-to-end | Missed faces | Incorrect labels | Duplicate assignments | Detection mean/median/p95 ms | Recognition mean/median/p95 ms | Image processing mean/median/p95 ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("A","B"):
        d=result["systems"][k]["synthetic_groups"]; kp=d["known_people"]; up=d["unknown_people"]
        pct=lambda x:"n/a" if x is None else f"{x:.4f}"
        def triplet(obj):
            return "/".join("n/a" if obj[z] is None else f"{obj[z]:.2f}" for z in ("mean_ms","median_ms","p95_ms"))
        lines.append(f"| {names[k]} | {d['detection_precision']:.4f}/{d['detection_recall']:.4f}/{d['detection_f1']:.4f} | {pct(kp['conditional_correct_identification_rate'])}/{pct(kp['end_to_end_identification_rate'])} | {pct(up['conditional_rejection_rate'])}/{pct(up['end_to_end_rejection_rate'])} | {pct(up['conditional_fpir'])}/{pct(up['end_to_end_fpir'])} | {pct(1-kp['conditional_correct_identification_rate'] if kp['conditional_correct_identification_rate'] is not None else None)}/{pct(1-kp['end_to_end_identification_rate'] if kp['end_to_end_identification_rate'] is not None else None)} | {d['missed_faces']} | {d['incorrect_identity_labels']} | {d['duplicate_identity_assignments']} | {triplet(d['detection_latency'])} | {triplet(d['recognition_latency'])} | {triplet(d['group_processing_latency'])} |")
    lines += ["","## Calibration and selection","",
              "Threshold and similarity margin are independently selected for each system and task (200-way LFW gallery; 50-way group gallery) on disjoint calibration identities, then frozen before test evaluation. LFW face selection is deterministic: largest box, then nearest image center, then top-left; identity labels and similarity scores are not used. Ambiguous (multi-face) cases and detection/embedding failures are recorded separately.","",
              "## Failures and artifacts","",
              "Per-image LFW/group predictions, detector caches, aligned face and embedding caches, calibration split and thresholds, synthetic annotations and annotated examples are included in this private output directory.","",
              "## Licensing and recommendation","",
              "SCRFD pretrained weights are restricted to non-commercial research unless separately authorized. AdaFace ONNX project code is MIT, but the release has no distinct checkpoint license and WebFace4M/model rights remain unresolved. Do not deploy System B before rights review.","",
              "The recommendation below is based on measured accuracy, CPU latency, model size and the licensing caveat. See the JSON for exact counts and failure records.","",
              f"**Recommendation:** {result.get('recommendation','pending')}",""]
    (out/"comparison_report.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
