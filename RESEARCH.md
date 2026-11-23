# FaceFind research and evaluation record

This file records measured results and clearly separates them from tests or
experiments that could not be run. WIDER FACE figures below come from saved
manifests under the ignored local path `data/private/evaluation/`; that data is
not committed.

## Application pipelines

The local assignment demo now defaults to SCRFD-10G-KPS + AdaFace IR-18. The
AdaFace matcher uses the independently calibrated 50-gallery threshold 0.26362
and top-score margin 0.07940. SCRFD uses score threshold 0.50, NMS threshold
0.40, downscales to a maximum dimension of 1,280, and pads its tensor to a
stride multiple of 32. AdaFace uses five-point alignment, BGR preprocessing,
and L2-normalized 512-dimensional embeddings.

YuNet + SFace remains a selectable fallback. Its original matcher values remain
recognition similarity 0.50, possible similarity 0.35, recognition margin
0.12, and possible-match margin 0.087; YuNet uses score threshold 0.50, NMS
threshold 0.30, and a maximum dimension of 1,280 pixels. The optional research
weights are not committed: SCRFD's pretrained weights are restricted to
non-commercial research under InsightFace's [model licensing terms](https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md).
AdaFace repository code is MIT, but the ONNX checkpoint's license and training
data rights remain unresolved. The research pipeline is not cleared for
commercial deployment. YuNet and SFace remain tracked with their license
notices in `models/face/`.

The local SCRFD checkpoint is the InsightFace SCRFD-10G five-keypoint
(`scrfd_10g_bnkps.onnx`) model; InsightFace's published pretrained model terms
restrict it to non-commercial research unless separately authorized. AdaFace
IR-18 ONNX came from the [yakhyo/adaface-onnx weights release](https://github.com/yakhyo/adaface-onnx/releases/tag/weights).
That repository's code is MIT-licensed, but the release does not provide a
separate checkpoint license or checksum; WebFace4M and checkpoint rights
remain unresolved. Neither checkpoint is distributed in this repository.

### Duplicate gallery identities in one group photo

`face_engine/pipeline.py:recognize_image` first matches each detected face
independently, then keeps the highest-scoring occurrence of any repeated
identity. Lower-scoring occurrences are marked possible if they meet the
possible similarity threshold and unknown otherwise; `conflict_with` points
to the retained face number. Equal scores are resolved by detection order.

The earlier Tagore/Einstein near-miss cannot be reconstructed from the
available repository: its exact image pair, embeddings, matcher results, and
scores were not retained. The regression fixture in
`tests/test_pipeline_conflicts.py` is synthetic and is not evidence about that
historical image. The current pipeline keeps the highest-scoring duplicate;
the AdaFace matcher rejects duplicate assignments as unknown.

## WIDER FACE custom detection evaluations

Both runs used the production YuNet detector at IoU 0.50 and the custom
confidence-ordered greedy one-to-one box matching implemented by
`scripts/evaluation/evaluate_widerface.py` (delegated to
`scripts/evaluation/widerface_common.py` in the modular layout).
These are sample precision/recall/F1 calculations, not official WIDER FACE AP.
Unreadable sampled images are excluded from the metric denominators and are
reported separately.

| Experiment | Images | Ground-truth faces | Predictions | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mixed density, seed 42 | 500 | 5,615 | 4,325 | 3,342 | 983 | 2,273 | 0.7727 | 0.5952 | 0.6724 |
| Exactly one annotated face, all eligible images | 1,118 | 1,118 | 1,387 | 1,004 | 383 | 114 | 0.7239 | 0.8980 | 0.8016 |

The stored manifest paths are `data/private/evaluation/widerface_mixed_500_seed42.json`
and `data/private/evaluation/widerface_single_face_seed42_all.json`. These
numbers were supplied as completed experiment results and are consistent with
the saved manifests. The single-face sample is restricted by valid annotation
count; this does not guarantee one visible or detectable face in each image.

## LFW and open-set evaluation

The saved end-to-end run at
`data/private/evaluation/final_end_to_end_comparison_20261124_01/` evaluated
the fixed 200-known/100-unknown LFW manifest (756 known and 557 unknown probes)
with thresholds calibrated on disjoint identities. Rank-1 identification was
98.15% for YuNet + SFace and 98.68% for SCRFD + AdaFace. Known correct-and-
accepted end-to-end rates were 95.11% and 98.28%, respectively. These are
results on the fixed LFW protocol, not a guarantee of real-world accuracy.
The paired synthetic group-photo experiment is reported separately and is not
evidence of real group-photo performance.

## Current app performance run

The actual FastAPI endpoints were measured over loopback using 52 prepared
named gallery references and the genuine local photo recorded in the private
input manifest. The group photo has no identity annotations; face counts are
operational observations only and must not be presented as accuracy. The
harness resolved ambiguous references through the preview/selection API by
choosing the largest detected face deterministically.

| Configuration | Gallery accepted / rejected / manual selection | Group faces detected / accepted identity labels | Warm API mean / median / p95 / max (ms) | Warm backend mean (ms) |
|---|---:|---:|---:|---:|
| SCRFD-10G-KPS + AdaFace IR-18 | 50 / 2 / 11 | 14 / 8 | 1,257 / 1,222 / 1,340 / 1,343 | 1,152 |
| YuNet + SFace | 52 / 0 / 10 | 13 / 6 | 553 / 552 / 581 / 582 | 463 |

Manual selections are included in the accepted gallery count. The two SCRFD
rejections had no detected face. Each configuration received one fresh
gallery upload, one first analysis, and five warm analyses. Gallery embeddings
were reused on repeated requests. No identity accuracy was assessed on this
photo. Full stage timings, per-reference outcomes, hashes, and input
provenance are in the ignored `final_app_performance_20261124_01/` directory.

The run used the existing local Python environments without installing
packages. Process/system memory could not be measured because `psutil` was not
available. Current focused regression checks passed 10/10 with Python's
`unittest`; syntax compilation also passed. This does not substitute for
testing on a clean clone with the optional model weights and dependencies.

## YuNet vs SCRFD-10G-KPS: 500-Image Detector Evaluation

### Objective and models

This experiment compares the existing YuNet face detector with SCRFD-10G-KPS
to measure whether the latter improves detection on group photographs and
images containing faces at different scales. YuNet uses the production
`face_detection_yunet_2023mar.onnx` model. SCRFD-10G-KPS uses the
`scrfd_10g_bnkps.onnx` five-landmark detector. This is a detector-only
comparison; SFace and the production recognition pipeline were not changed.

### Dataset and methodology

The evaluation used all 500 image paths from the fixed, seed-42 manifest
`data/private/evaluation/widerface_mixed_500_seed42.json`. The runner preserved
that exact image list for both detectors and checked each selected image
against the WIDER FACE validation annotations. There were 5,615 annotated
faces across the sample.

Both detectors used common comparison preprocessing: aspect-preserving resize
to a maximum dimension of 1,280 pixels, `INTER_AREA` when downsampling, and
bottom/right padding to a multiple of 32. This preprocessing differs from the
historical YuNet evaluator, which did not use this padding. YuNet used a 0.50
confidence threshold and 0.30 NMS threshold; SCRFD used 0.50 confidence and
0.40 NMS. Matching used confidence-ordered greedy one-to-one box assignment at
IoU >= 0.50. SCRFD was evaluated using CPU execution (`ctx_id=-1` in the
comparison runner). The saved JSON does not include the active provider name,
so this CPU attribution is based on the evaluator's initialization setting.

These are custom precision, recall, and F1 metrics at IoU 0.50. They are **not
official WIDER FACE AP results**, and this fixed 500-image sample is **not the
full official benchmark**.

### Aggregate results

| Detector | Images | Ground-truth faces | Predictions | TP | FP | FN | Precision | Recall | F1 | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YuNet (common comparison preprocessing) | 500 | 5,615 | 4,325 | 3,338 | 987 | 2,277 | 0.7718 | 0.5945 | 0.6716 | 63.6016 | 57.8081 | 105.1260 |
| SCRFD-10G-KPS (CPU) | 500 | 5,615 | 3,682 | 3,519 | 163 | 2,096 | 0.9557 | 0.6267 | 0.7570 | 408.1040 | 369.4971 | 614.2754 |

There were no unreadable images or recorded preprocessing, detector,
prediction-validation, or matching exceptions. For each detector, TP + FN is
5,615 and TP + FP equals the prediction count. Per-image counts sum to the
saved aggregates.

SCRFD produced 824 fewer false positives, an 83.49% reduction relative to
YuNet, and 181 more true positives. Recall increased by 3.22 percentage
points, precision by 18.39 points, and F1 by 8.54 points. This sample therefore
shows a substantial precision improvement and a smaller recall improvement.

### Recall by annotated face height

| Ground-truth face height | Ground-truth faces | YuNet TP | YuNet recall | SCRFD TP | SCRFD recall | Recall change |
|---|---:|---:|---:|---:|---:|---:|
| <= 20 px | 2,931 | 1,154 | 0.3937 | 1,287 | 0.4391 | +4.54 pp |
| > 20 to 40 px | 1,331 | 989 | 0.7431 | 1,016 | 0.7633 | +2.03 pp |
| > 40 px | 1,353 | 1,195 | 0.8832 | 1,216 | 0.8987 | +1.55 pp |

SCRFD had higher recall in all three predeclared height groups. Small-face
recall remained the weakest for both detectors: SCRFD detected 1,287 of 2,931
faces at or below 20 pixels in height.

### CPU latency

The report describes these as cold-inclusive per-image latency measurements.
They include comparison resize and padding, detector preprocessing, inference,
postprocessing, and any lazy runtime/session initialization incurred during
the measured call. They exclude disk image loading, prediction validation,
IoU matching, and report writing. The first inference is included; these are
not steady-state latency claims.

On this run, SCRFD mean latency was about 6.42 times YuNet's; median latency
was about 6.39 times higher. Its stronger sample detection metrics came with a
substantial CPU speed cost.

### Interpretation and limitations

The saved results support SCRFD-10G-KPS as a promising research comparison for
reducing false positives while modestly improving recall on this fixed WIDER
FACE sample. They do not establish statistical significance, generalization
to other photo distributions, recognition accuracy, or end-to-end application
quality. The comparison is custom IoU evaluation rather than official WIDER
FACE AP, and YuNet's result uses the runner's common padded preprocessing
rather than its historical preprocessing.

The SCRFD pretrained weights are restricted to non-commercial research under
InsightFace's [model licensing terms](https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md).
Commercial deployment should not be considered without appropriate licensing.
This detector-only evaluation is distinct from the later app integration. The
current local demo selects SCRFD + AdaFace by default, with YuNet + SFace kept
as a fallback; the WIDER FACE detector sample alone does not establish
recognition accuracy or real-world group-photo performance.

The saved report is
`data/private/evaluation/widerface_yunet_scrfd10g_kps_full500_seed42/detector_comparison.json`.
