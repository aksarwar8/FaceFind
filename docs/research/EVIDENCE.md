# Sanitized research evidence index

The run identifiers below refer to retained local artifacts. This index carries no raw images, embedding vectors, per-person records or filesystem locations for private inputs. The quantitative tables and interpretation are in [RESEARCH.md](../../RESEARCH.md).

| Run identifier | Retained evidence inspected | Claims supported and boundaries |
|---|---|---|
| `widerface_mixed_500_seed42` | Evaluation manifest with aggregate and sample counts | Production YuNet, 500 images, custom IoU counts. |
| `widerface_single_face_seed42_all` | Filtered manifest and aggregate counts | 1,118 images with one eligible annotation. |
| `widerface_single_face_diagnostic_seed42_01` | Diagnostics JSON and per-image box comparisons | Failure-enriched 20-image diagnostic. No global prevalence estimate. |
| `widerface_yunet_scrfd10g_kps_pilot25_seed42`, `widerface_yunet_scrfd10g_kps_full500_seed42` | Separate detector comparison JSON files | Pilot and full two-way fixed-sample runs. Timing differs from the later three-way run. |
| `widerface_retinaface_full500_seed42` | Detector comparison JSON and settings | Three-detector count, timing, and size-stratified recall on one cohort. |
| `yolov8n_vs_scrfd_widerface_full3226_20261123` | Partial report, results JSON and preserved prediction rows | Shared ordered 1,500-image comparison. Full dataset incomplete and latency unavailable. |
| `open_set_run_20261123`, `open_set_run_20261123_v2` | Reports and reproducibility manifests | Gallery mismatch, preflight correction, known/unknown denominators and margin sensitivity. |
| `recognition_comparison_sface_arcface_adaface_full_20261123_01` | Report, results JSON, calibration split and frozen thresholds | Shared detector, fixed test split, recognition metrics, EER/AUC and CPU cost. Separate from official LFW pairs. |
| `final_end_to_end_comparison_20261123` | Status and reproducibility manifest | Blocked before execution. Carries no usable benchmark metrics. |
| `final_end_to_end_comparison_20261124_01` | Report, results JSON, calibration, frozen parameters and composite manifests | Completed combined LFW and controlled composite comparisons. Runtime AdaFace settings correspond to the 50-way experiment. |
| `final_app_performance_20261124_01` | Benchmark report, aggregate JSON and input provenance | Real FastAPI loopback workload. Genuine group photo without identity ground truth. |
| `reference_detection_diagnostic_20261124` | Raw detector scores, controlled scales, annotated comparisons and adaptive gallery validation | Reproduced the 17-photo discrepancy, measured retry time, and exercised the adaptive path on 23 additional inputs. Limited cohort, no recalibration. |
| `reference_false_positive_diagnostic_20261124` | Crop inspection, raw detector scores, scale and NMS controls | Wall recess scored .5330 and had zero overlap with the true face at .8286. |
| `smart_selection_validation` | Strict and Smart gallery outcomes, annotated smaller-target controls and correction check | Required reviews fell from three to one in the 17-photo folder. All three automatic choices were wrong in the designated smaller-target cases where Smart selected. |
| `release_verification_20261124` | Guarded real HTTP workflow, clean-install status and actual export validation | See the [release report](../verification/RELEASE.md). Separate from historical performance benchmarking. |
| `facefind_milestone2_verification_20261124_01` | Verification report, results, adapter check and environment context | Workflow smoke, calibration provenance and cache behavior. See the [milestone record](../verification/MILESTONES.md). |

The [machine-readable aggregates](benchmark-aggregates.json) contain selected detector and FastAPI benchmark measurements. The code supports the implemented resize and padding policies. The original single-photo before/after measurements, historical identity near-miss, official LFW pairs report, and reason for the YOLO run interruption were unavailable. Local hashes and bytes are in the [model rights register](../../models/face/MODEL_LICENSES.md).

## Final methodology clarification

The recognition-only calibration split and its failures are backed by the saved comparison results and frozen threshold files. The combined run records a separate calibration/test identity overlap check, per-pipeline probe counts, measured failures, and separate 200-way and 50-way frozen operating points. The synthetic group evaluator matches detections to ground truth before calculating person-level identity metrics. Unmatched detections affect detection precision. Duplicate counts precede production conflict resolution. The saved identity split does not test overlap with model pretraining data. [Calibration and denominator summary](../../RESEARCH.md#calibration-and-denominator-summary).

## Evaluation Studio and documentation figures

| Evidence | Verified conclusion |
|---|---|
| `evaluation_studio_20261124` per-group Chrome exports and synthetic answer keys | 60 gallery identities, five composites, 57 annotated faces. 34 correct known, 22 correct unknown, one known rejection, and TP/FP/FN 57/0/0. |
| Saved calibration split and frozen Studio gallery | Identity-name matches demonstrate overlap with the 50-way calibration gallery and broader LFW pools. Only intersection counts are released. See the [overlap audit](../../RESEARCH.md#identity-overlap-audit). |
| Recognition-only evaluator and saved counts | The legacy `end_to_end_rejection_rate` is a complement of FPIR. Correct unknown rejections require a valid evaluated result. The report gives attempted and evaluated denominators. |
| Historical warm application performance JSON | Unaccounted backend and API-minus-backend means are arithmetic residuals using the same five requests per pipeline. Their internal composition is unmeasured. |

The six [figures](figures/) are generated by [render_figures.py](render_figures.py) using only [benchmark-aggregates.json](benchmark-aggregates.json). To reproduce them, run the script from a separate Python environment with Matplotlib 3.10.7 and NumPy 1.26.4. The application dependencies and saved benchmark outputs are unchanged.
