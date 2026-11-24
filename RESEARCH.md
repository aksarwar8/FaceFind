# FaceFind research and engineering decisions

The evidence supports a two-pipeline application. SCRFD + AdaFace is the preferred local research/demo configuration because it improved detection and identification in controlled comparisons. YuNet + SFace provides a smaller, faster, bundled path. The [evidence index](docs/research/EVIDENCE.md) maps each result to its originating run and documents missing records.

## 1. Problem and baseline

Detection localizes faces and estimates landmarks. Recognition operates on an aligned face and retrieves the closest gallery embedding. In an open-set system, every probe has a nearest candidate even when the person is absent. FaceFind uses a minimum cosine score and a minimum gap between the first and second candidates. Increasing gallery size introduces more opportunities for a high impostor match and can shrink the top-two margin. Thresholds depend on model, gallery size and data distribution. They do not turn similarity into a confidence probability.

The baseline paired OpenCV Zoo YuNet with SFace. Its small detector, CPU speed, five-point landmarks and OpenCV deployment made it suitable for the first complete local workflow. Pose, occlusion, blur, tiny faces and ambiguous enrollment images can reduce either detection or recognition quality. Separate detector, recognition and combined evaluations identify where failures occur.

### Resolution sensitivity

The historical scaling script compares a reference at its source dimensions and at a resized maximum dimension. The exact historical before/after counts and alleged fragmented detections could not be verified from a saved output. The implemented YuNet policy downsizes proportionally only when the long edge exceeds 1,280 pixels and remaps boxes and landmarks to source coordinates. The larger-image behavior is a documented investigation, with the exact historical image claims left open. SCRFD additionally pads the resized detector input to stride 32.

## 2. Detector experiments

### Protocol and initial YuNet measurements

The [WIDER FACE dataset](https://shuoyang1213.me/WIDERFACE/) provides validation images and annotated face boxes. The evaluator reads the image path, box count and annotation rows, converts valid boxes to a common coordinate convention, and matches predictions in descending detector score to the highest-IoU unmatched eligible annotation at IoU ≥ 0.50. Matches are TP, unmatched predictions FP and unmatched annotations FN. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. Exclusions such as unreadable images are reported separately. This is a custom box protocol. Official WIDER FACE Easy/Medium/Hard AP uses a different evaluation procedure.

| Production YuNet cohort | Images | Annotated faces | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mixed density, fixed seed 42 | 500 | 5,615 | 3,342 | 983 | 2,273 | .7727 | .5952 | .6724 |
| All images with exactly one eligible annotated face | 1,118 | 1,118 | 1,004 | 383 | 114 | .7239 | .8980 | .8016 |

A single eligible annotation does not establish that only one person or face is visible. The seed-42 failure audit selected 10 images with misses and 10 with additional predictions from that cohort. The diagnostic rerun recorded 8 TP, 22 FP and 12 FN across the selected images. This failure-enriched selection cannot estimate overall failure prevalence. Low IoU can count a localized but displaced prediction as both an FP and an FN. Extra detections can also include real but unannotated faces. The saved artifact has coordinate evidence and examples. It does not establish separate blur, occlusion, or boundary-proximity effects. The controlled face-height results below quantify the size association directly.

### Controlled three-detector comparison

The compared detectors received the same 500 WIDER FACE images and 5,615 annotated faces. The comparison uses common resize and bottom/right stride padding, which differs from the original standalone YuNet protocol. CPU timing includes preprocessing and detection, including any first-use overhead, and excludes file loads and IoU matching.

| Detector | TP | FP | FN | Precision | Recall | F1 | Mean / median / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| YuNet | 3,338 | 987 | 2,277 | .7718 | .5945 | .6716 | 99.71 / 92.65 / 171.55 |
| SCRFD-10G-KPS | 3,519 | 163 | 2,096 | .9557 | .6267 | .7570 | 526.14 / 483.75 / 891.80 |
| RetinaFace MobileNet 0.25 | 1,881 | 407 | 3,734 | .8221 | .3350 | .4760 | 106.59 / 94.05 / 185.96 |

| Annotated height | Faces | YuNet recall | SCRFD recall | RetinaFace recall |
|---|---:|---:|---:|---:|
| ≤20 px | 2,931 | .3937 | .4391 | .0389 |
| >20–40 px | 1,331 | .7431 | .7633 | .4568 |
| >40 px | 1,353 | .8832 | .8987 | .8566 |

SCRFD improved recall in each size bucket and produced far fewer unmatched predictions than YuNet, at higher CPU cost. Even SCRFD missed most annotations in the smallest bucket. This RetinaFace export recovered few tiny faces, so its results did not justify selecting it for the project. A separate two-way pilot and 500-image run exist. Their measured latencies are kept separate from this table. See [checkpoint sizes and rights](models/face/MODEL_LICENSES.md).

### Partial YOLOv8n-Face comparison

The 3,226-image run was interrupted. The saved report does not establish the reason. The harness preserved 1,701 YOLO and 1,709 SCRFD prediction rows. It retained the first 1,500 ordered rows common to both files, verified identical unique image paths, and evaluated 25,263 boxes under the custom IoU protocol.

| Detector | TP | FP | FN | Precision | Recall | F1 | Recall ≤20 / >20–40 / >40 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n-Face | 10,623 | 245 | 14,640 | .9775 | .4205 | .5880 | .1366 / .6403 / .8807 |
| SCRFD-10G-KPS | 16,212 | 748 | 9,051 | .9559 | .6417 | .7679 | .4538 / .8268 / .8992 |

The saved prediction files do not contain comparable per-image timings. No speed ranking is supported. This cohort is independent of the earlier 500-image sample.

## 3. Recognition investigation

### Models and preprocessing

[SFace](https://arxiv.org/abs/2205.12010) uses a sigmoid-constrained hypersphere objective and the bundled OpenCV model produces 128-dimensional vectors. [ArcFace](https://arxiv.org/abs/1801.07698) uses an additive angular margin during training. The benchmarked InsightFace ResNet100 checkpoint produces 512-dimensional vectors. [AdaFace](https://arxiv.org/abs/2204.00964) adapts its training margin using feature norm as a quality proxy. The benchmarked IR-18 export produces 512 dimensions. This training objective does not give the runtime a calibrated confidence score.

The comparison used shared YuNet detections with model-specific alignment/input normalization. The deployed SFace adapter uses OpenCV's five-point `alignCrop` and native recognition model. AdaFace estimates a five-point similarity transform to 112 × 112, normalizes BGR input and L2-normalizes embeddings. Checkpoint byte sizes and rights are recorded separately.

### Pair verification and corrected open-set identification

A standalone evaluator implements the official LFW pairs workflow, but no completed pairs-protocol output was located. ROC-AUC and EER below come from the recognition benchmark's verification component. They are not official LFW ten-fold results.

The open-set experiment uses one reference per known identity, held-out known probes, and identity-disjoint unknown probes. The early evaluator required a single detected face for use. The first run chose 200 known identities, but enrolled only **146** while still evaluating probes from the original known list. That made some known probes impossible to identify by construction. Version 2 preflighted references, attempted alternatives and used reserves to restore a consistent 200-identity gallery. It replaced 54 selections, excluded 65 candidates, and verified that the enrolled identities and known probe identities agree. The known probe totals changed from 696 to 756, so v1 and v2 rates describe different cohorts.

| V2 possible-match margin policy | Known correct / evaluated | Unknown rejected / evaluated | Unknown false IDs / evaluated |
|---|---:|---:|---:|
| Retained legacy policy | 559 / 578 | 424 / 442 | 18 / 442 |
| Lower possible margin | 567 / 578 | 383 / 442 | 59 / 442 |
| Higher possible margin | 528 / 578 | 440 / 442 | 2 / 442 |

This experiment counts possible labels in the identification totals. Of 756 known probes and 557 unknown probes, 178 known and 115 unknown probes failed evaluation; those are not correct rejections. The more permissive possible margin accepted more known identities at the cost of substantially more false unknown assignments.

### Controlled SFace, ArcFace and AdaFace comparison

The saved run used one shared gallery of 200 known identities and 1,313 fixed test probes (756 known, 557 unknown), with 100 unknown identities. The independent calibration set selected 200 known and 40 unknown identities. Only 147 known calibration references passed the single-face policy. The saved threshold summaries include 445 genuine scores, 141 unknown maximum scores and 240 calibration failures per model. Each model's cosine threshold was frozen before test scoring. Rank-1 describes the nearest identity before thresholding. Correct acceptance requires the correct identity above threshold.

| Model | Rank-1 conditional / end-to-end | Correct acceptance conditional / end-to-end | Unknown rejection conditional | FPIR conditional / end-to-end | FNIR conditional / end-to-end | EER / ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SFace | .9896 / .7566 | .9654 / .7381 | .9751 | .0249 / .0197 | .0346 / .2619 | .0059 / .9988 |
| ArcFace | .9983 / .7632 | .9965 / .7619 | 1.0000 | .0000 / .0000 | .0035 / .2381 | .0017 / .9990 |
| AdaFace | .9983 / .7632 | .9965 / .7619 | .9774 | .0226 / .0180 | .0035 / .2381 | .0017 / .9989 |

| Model | Frozen threshold | CPU embedding mean / median / p95 ms | Model MB, decimal | Failed test probes |
|---|---:|---:|---:|---:|
| SFace | .423053 | 23.21 / 22.48 / 31.12 | 38.7 | 293 |
| ArcFace | .368590 | 789.18 / 789.07 / 903.43 | 260.7 | 293 |
| AdaFace | .291446 | 55.64 / 54.77 / 65.59 | 96.1 | 293 |

Conditional rates use processed probes. FPIR is the false positive identification rate for unknowns. FNIR is the false negative identification rate for knowns, including wrong identities and non-acceptance. End-to-end known metrics include failures as misses. End-to-end FPIR divides by all unknown probes and counts only false identifications in the numerator. Its complement would include processing failures, so use the conditional unknown rejection rate and the explicit failure count to interpret performance. The recognizer comparison used Python 3.11.9, OpenCV 4.12.0, and ONNX Runtime 1.32.0 with the CPU execution provider. ArcFace achieved the lowest observed FPIR on this sample but carried the highest CPU latency and file size. AdaFace matched ArcFace's known acceptance results with markedly lower CPU embedding latency.

## 4. Combined pipeline evaluation and deployment calibration

The completed November 24 run changed from requiring a single detection to deterministic face selection on LFW: largest box, then nearest center, then top-left. It records ambiguous probes independently and does not use identity labels or similarity to select a face. Calibration identities are disjoint from test identities. It preflighted 200 usable known calibration identities and froze separate threshold/margin pairs for the 200-way LFW and 50-way group tasks. This procedure and the recognition-only procedure have different failure denominators despite sharing the test protocol. The earlier November 23 combined attempt was blocked before execution and contributes no result.

| System | 200-way threshold / margin | 50-way threshold / margin |
|---|---:|---:|
| YuNet + SFace | .427028 / .026805 | .385532 / .066137 |
| SCRFD + AdaFace | .286670 / .062926 | .263616 / .079397 |

The deployed AdaFace rule uses **0.26362 similarity and 0.07940 margin**, matching the group experiment after rounding. The deployed fallback retains similarity **0.50** and margin **0.12**, with a possible branch at **0.35** and **0.087**. The experimental SFace rules stayed in research. The saved grid and frozen values describe this calibration split and do not prove transfer to other cohorts.

### Calibration and denominator summary

| Protocol | Calibration identities and gallery | Evaluated calibration probes | Test identities and gallery | Test probes and failures |
|---|---|---|---|---|
| Recognition-only LFW | 200 known and 40 unknown sampled. 147 known calibration references passed the single-face policy. | 445 genuine scores, 141 unknown maximum scores and 240 failed calibration probes per model. | 200 known gallery identities and 100 unknown identities. | 756 known and 557 unknown. Of these, 578 known and 442 unknown were evaluated. Failures were 178 known and 115 unknown. |
| Combined LFW | 200 usable known calibration identities and 40 unknown identities. 200-way gallery. | 700 known and 184 unknown per pipeline, zero recorded calibration failures. | 200 known gallery identities and 100 unknown identities. | 756 known and 557 unknown evaluated per pipeline, zero recorded evaluation failures. |
| Combined synthetic groups | 50-way gallery using the separate combined calibration pool. | 201 known and 184 unknown per pipeline, zero recorded calibration failures. | 50 enrolled identities. The composites contain 50 known and 50 unknown face instances. | YuNet matched 46 known and 45 unknown boxes, and SCRFD matched 50 known and 48 unknown. Neither pipeline recorded an embedding failure on the matched boxes. |

The saved recognition-only and combined evaluations report empty calibration/test identity intersections. This separation does not establish whether the model pretraining datasets contained overlapping identities. Frozen thresholds and margins are listed with their respective protocols above. See the [source inventory](docs/research/EVIDENCE.md).

### Fixed LFW cohort

| System | Rank-1 | Known correct+accepted | Unknown rejected | FPIR | FNIR | Ambiguous probes |
|---|---:|---:|---:|---:|---:|---:|
| YuNet + SFace | .9815 | .9511 | .9785 | .0215 | .0489 | 293 |
| SCRFD + AdaFace | .9868 | .9828 | .9820 | .0180 | .0172 | 266 |

Both pipelines used 756 known and 557 unknown probes with zero gallery failures. Saved conditional and end-to-end rates are equal on this run. The ambiguity counts are detections with multiple candidate faces, not processing failures.

### Controlled group composites

Ten composites held 100 faces, evenly split between known and unknown people. Crops came from held-out probe images; enrollment crops were excluded. Transformed Haar boxes supplied composite annotations. Face size, blur, lighting and output resolution varied. The composites provide controlled stress cases and do not measure accuracy on genuine photos.

| System | Detection P / R / F1 | Known correct conditional / end-to-end | Unknown rejection conditional / end-to-end | FPIR conditional / end-to-end | FNIR conditional / end-to-end | Missed | Image mean / median / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| YuNet + SFace | .9785 / .9100 / .9430 | .7826 / .7200 | .9333 / .8400 | .0667 / .0600 | .2174 / .2800 | 9 | 653.31 / 647.72 / 850.22 |
| SCRFD + AdaFace | .9608 / .9800 / .9703 | .9000 / .9000 | .9583 / .9200 | .0417 / .0400 | .1000 / .1000 | 2 | 1,678.53 / 1,640.30 / 2,589.33 |

Identity metrics use detections matched to ground-truth boxes. Unmatched detections count as detection false positives. Any accepted identity on an unmatched box is outside the person-level FPIR numerator. The group conditional denominators are matched known or unknown boxes, unlike the recognition-only denominators based on evaluated probes. The group end-to-end denominators include every annotated person in the corresponding known or unknown class. The comparison records duplicate assignments before the production duplicate-conflict pass. A missed detection, an embedding failure, and a processed match rejected by the decision rule are distinct outcomes.

Combined checkpoint size was about **38.93 MB** for YuNet + SFace and **113.03 MB** for SCRFD + AdaFace, calculated from local file sizes. SCRFD + AdaFace traded lower synthetic detection precision and greater latency for higher recall and better known identification in this fixed experiment. Real deployment clearance was a separate question from choosing a research/demo default.

## 5. Actual FastAPI application performance

The local harness started Uvicorn and issued HTTP requests over loopback. Both pipelines received the same 52 prepared references and one genuine group photo, with one first analysis and five warm requests per model. For timing only, the harness resolved ambiguous references by selecting the largest detected face. The production UI leaves that choice to the user. The CPU model and memory were not recorded in the report, and resource sampling was unavailable because `psutil` was absent.

| Pipeline | Init ms | Enrolled / rejected / manual¹ | Gallery request ms | Detected / assigned | First API ms | Warm API mean / median / p95 / max ms | Warm backend mean / median / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| SCRFD + AdaFace | 1,971.0 | 50 / 2 / 11 | 11,105.0 | 14 / 8 | 1,283.0 | 1,257.4 / 1,222.3 / 1,340.3 / 1,343.4 | 1,151.9 / 1,159.6 / 1,226.0 |
| YuNet + SFace | 143.1 | 52 / 0 / 10 | 2,225.6 | 13 / 6 | 645.4 | 552.7 / 551.6 / 581.2 / 582.1 | 462.6 / 451.9 / 501.9 |

¹ Manual selections are included in final enrollment. Gallery client time includes the upload and the manual-selection requests. Both caches reported zero gallery embedding time during repeated analysis.

| Warm stage mean, ms | Detect | Align | Embed | Match | Annotate and render |
|---|---:|---:|---:|---:|---:|
| SCRFD + AdaFace | 405.1 | 15.1 | 505.4 | 3.3 | 96.1 |
| YuNet + SFace | 65.0 | 5.2 | 181.7 | 2.4 | 105.8 |

The group photo was unannotated. The assigned counts describe application output. No identity correctness can be derived, and enrollment ended with different galleries. The reported latency includes the conditions of this one-photo benchmark.

## 6. Engineering resolution and final decision

- **SCRFD input shape.** The adapter rounds the resized input dimensions up to the stride multiple before inference. Bottom/right padding preserves the origin. It clips boxes and reverses the resize scale to recover source coordinates. A regression test checks the padded tensor shape and returned geometry. The earlier interrupted output does not independently establish every original failure symptom.
- **Ambiguous references and gallery safety.** The application queues multi-face references for explicit selection. Failed replacement preserves the working gallery. Each pipeline owns separate vectors, pending references and thumbnails. Switching validates the new bundle before activation.
- **Unknown and duplicate handling.** Unknown results carry a null assigned identity. A post-match pass retains the highest-scoring repeat of an identity, with first detection breaking ties. AdaFace demotes the others to Unknown. SFace can leave a possible candidate. A private historical near-miss lacks retained inputs and scores; the synthetic regression verifies the mechanism.
- **Runtime and dependencies.** An earlier interpreter pointed to an unavailable Windows base executable. The unified requirements now describe one Python 3.11 environment. The current NumPy, ONNX and ml_dtypes pins and successful imports were verified. No preserved resolver transcript establishes the exact old NumPy/ml_dtypes failure sequence. InsightFace's declared OpenCV package is omitted to keep one `cv2` provider. The Milestone 2 adapter check verified a finite normalized 512-dimensional embedding after updating the scikit-image alignment call.
- **Startup and selection.** An unavailable optional model previously threatened startup. The current app tries the alternate pipeline and reports the reason. A failed switch preserves the working bundle. This made the better measured research system a practical preferred demo while retaining a fast, bundled path for fresh checkouts.

### Reference enrollment scale sensitivity

The real 17-photo reference folder exposed a failure mode outside the detector benchmark. At the normal 1,280-pixel cap, SCRFD produced no candidates above 0.50 for three close-up faces that YuNet detected and visual inspection confirmed. The three SCRFD peak scores were .1202, .0750, and .2891. The candidate lists were empty before NMS or clipping. Rotation and the legacy decoder did not recover the faces; reducing the content scale recovered all three, even on a fixed-size detector tensor. The experiment establishes scale sensitivity for those photos without identifying a universal face-size boundary.

The application retains the 1,280-pixel primary pass. Gallery enrollment retries SCRFD at 640 pixels only when the primary pass has no valid boxes. Both passes retain the existing score and NMS settings. Coverage rose from **14/17 to 17/17** in the original folder. Retry detections are mapped to the working image for AdaFace alignment. The table used manual review for every multi-face reference. Analyze uses the original one-pass behavior.

| Validation set | Direct enrollment | Manual selection | Rejected | Retry attempts / detections recovered | Retry total / median |
|---|---:|---:|---:|---:|---:|
| Original 17 references | 14 | 3 | 0 | 3 / 3 | 714.0 / 228.8 ms |
| Separate 23-image validation set | 9 | 9 | 5 | 11 / 6 | 2,419.2 / 203.8 ms |

On the original set, all 14 nonempty primary results matched the saved boxes, landmarks and scores exactly; the three recovered references matched the previously visually verified 640-pixel detections. The separate set included six independently selected LFW portraits, six modified portraits covering large framing, blur and low light, three two-face composites, two multi-person composites, one genuine group photo, and five procedural negatives. The retry recovered three degraded close-ups and detected both faces in each of the three two-face composites, which all entered manual review. All five negatives were rejected. Visual inspection checked recovered boxes and landmark placement. All 23 automatically enrolled vectors across both sets were finite, 512-dimensional, and unit-normalized. Retry timings measure the additional detector pass in one local CPU run and exclude decoding and embedding.

The production matching threshold and margin remain unchanged. The checks establish that recovered landmarks can produce valid AdaFace embeddings and that the tested cases preserve enrollment semantics. They do not establish calibration transfer or a false-positive rate on a representative cohort. Broader validation would require annotated real references, hard negatives and disjoint known/unknown probes using retry-enrolled galleries. Private experiment outputs, timings and annotated contact sheets are retained under the reference detection diagnostic run.

### Smart reference auto-selection

A separate diagnostic confirmed that a second SCRFD box at .5330 enclosed a brick-wall recess. The real face scored .8286 and the two boxes had zero IoU. NMS correctly collapsed overlapping proposals for the real face and could not suppress the separate false positive by overlap. YuNet returned one face at .9138 on the same decoded image. The experiment supported exposing both crops for review while preserving detector thresholds.

Local validation compared **Always Review Multiple Faces** and **Smart Auto-Selection** through the gallery upload handler using SCRFD-10G-KPS + AdaFace on CPU. The original 17 files included the confirmed brick-wall false positive. The additional set contained one genuine local group photo, two multi-person composites, three similarly sized two-person composites, three unequal-size two-person composites and six genuine two-person photographs from the locally available WIDER FACE validation split. For the six WIDER FACE stress cases, the smaller annotated face was designated as the intended target before inference. Both annotated faces had separate matching detections in every case.

| Validation set | Required reviews, always → smart | Multi-face auto-selections | Incorrect auto-selections |
|---|---:|---:|---:|
| Original 17, including the brick-wall case | 3 → 1 (67% reduction) | 2 | 0 of 2 |
| Genuine local group, 2 multi-person composites and 3 unequal-size composites | 6 → 6 | 0 | 0 |
| Similar-size pairs, area ratios 1.03–1.21 | 3 → 3 | 0 | 0 |
| Genuine two-person photos with the smaller face designated as target | 6 → 3 (50% reduction) | 3 | 3 of 3 |

In the brick-wall photo the intended face had a 22.18× area ratio and a 0.8286 score. The wall candidate scored 0.5330 and remained available in the optional review. The other original auto-selection picked the foreground subject at a 16.03× area ratio and a 0.9005 score. A real background face remained visible in that review. No photos were rejected under either setting. Both settings recovered the same three original references through the existing SCRFD retry. Group analysis returned 14 detections under each setting on the same local group photo.

The test also replaced an incorrect automatic choice with the smaller intended face and verified the embedding and gallery revision changed. A reduction in required reviews counts wrong choices as automatic too. The original 17 measure the local folder only, and the smaller-target cases were deliberately selected to stress the heuristic. Face size and detection score cannot establish which person the uploader intends to enroll.

## 7. Dataset and rights decisions

| Dataset | Evidence of use | Rights and scope |
|---|---|---|
| WIDER FACE | Detector samples and failure inspection | Use under dataset terms. No image redistribution by FaceFind. |
| LFW | Open-set, recognition comparison, combined evaluation and group crop sources | Use under the dataset's terms. Image and person rights are separate from model code. |
| Local references and group photo | Application timing and workflow smoke tests | Restricted to local artifacts. No public image or identity manifest accompanies this report. |
| Synthetic group composites | Controlled geometry and matching experiment | Derived from held-out LFW probes. No public composites supplied here. |
| CelebA, Open Images, CFP-FP, AgeDB | No verified use in the FaceFind experiments | No adoption or rejection rationale independently established. |
| SoloFace2 | No verified local use or original dataset-card or conversation evidence for the reported circularity concern | Treat that concern as an evidence gap. |

The [model rights register](models/face/MODEL_LICENSES.md) records source-code terms, checkpoint provenance, local hashes and distribution scope. A permissive repository license does not settle rights to all checkpoints, training data or evaluation images. SCRFD weights have a non-commercial research restriction and AdaFace checkpoint rights remain unresolved. Neither is included for redistribution or cleared here for commercial use.

## 8. Limitations and references

The detector protocol is custom and the YOLO comparison partial. LFW is a different distribution from small faces in group photographs, and the synthetic groups cannot substitute for annotated genuine groups. Thresholds may change with gallery size and capture conditions. CPU measurements describe these local runs and cannot establish GPU performance or memory behavior. Final desktop UI validation remains manual. See [verification](docs/verification/MILESTONES.md).

- Detection: [WIDER FACE](https://shuoyang1213.me/WIDERFACE/), [SCRFD paper](https://arxiv.org/abs/2105.04714), [RetinaFace paper](https://arxiv.org/abs/1905.00641), [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
- Recognition: [SFace paper](https://arxiv.org/abs/2205.12010), [ArcFace paper](https://arxiv.org/abs/1801.07698), [AdaFace paper](https://arxiv.org/abs/2204.00964), [LFW paper](https://people.cs.umass.edu/~elm/papers/Huang_eccv2008-lfw.pdf).
- Checkpoints and rights: [InsightFace licensing](https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md), [SFace distribution](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), [AdaFace ONNX release](https://github.com/yakhyo/adaface-onnx/releases/tag/weights).
