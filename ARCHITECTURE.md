# FaceFind system design

**Application baseline:** [`9734b203f524670b695e85b0ea2088f4c6794517`](https://github.com/aksarwar8/FaceFind/commit/9734b203f524670b695e85b0ea2088f4c6794517)

[Context](#system-context-and-requirements) · [Components](#components-and-boundaries) · [Decisions](#architecture-decision-records) · [Enrollment](#enrollment) · [Inference](#group-photo-inference) · [State](#state-management-and-concurrency) · [Evaluation](#evaluation-studio) · [API](#api-contracts) · [Security](#security-and-privacy) · [Reproducibility](#reproducibility) · [Archify](#archify-artifact)

## System context and requirements

FaceFind accepts reference images, maintains a gallery, analyzes a group photo, and lets an evaluator score the frozen inference against independent annotations. The deployment boundary is a desktop browser and one FastAPI process on the same workstation. The supported workflows are observable through model status, review state, structured results, and exports.

| Requirement | Architectural response |
|---|---|
| Run the recognition demo on a CPU without sending face data to a hosted service | Loopback API, local browser assets and local ONNX-backed model adapters |
| Remain runnable without separately supplied research checkpoints | Bundled YuNet + SFace, automatic fallback and explicit pipeline selection |
| Preserve enrollment intent and report ambiguous references | Strict review, an optional dominant-face heuristic and correction |
| Keep inference results reproducible enough to evaluate | Per-analysis provenance and a frozen copy in the browser |
| Bound shared state and uploads | Versioned galleries, serialized API mutations and resource limits |
| Produce evidence that can be assessed independently | Explicit metrics, annotation provenance, separate exports and an auditable diagram |

~~~mermaid
flowchart LR
    O[Operator] --> B[Browser workspace]
    B <-->|Loopback HTTP| A[FastAPI worker]
    A --> D[Decode and run the active CPU bundle]
    A <--> G[Pipeline-local galleries in process memory]
    D --> R[Structured result and annotated JPEG]
    R --> B
    R --> F[Frozen browser snapshot]
    T[Independent truth file or manual correspondence] --> E[Evaluation Studio in browser]
    F --> E
    E --> X[Evaluation exports]
~~~

[Open the genuine Archify artifact](docs/architecture/facefind.html). Mermaid diagrams in this document describe specific flows and complement the Archify overview.

## Components and boundaries

| Component | Responsibility |
|---|---|
| [app.py](app.py) | FastAPI lifespan, model startup, process caches, static UI delivery and serialization of local API mutations. |
| [api/routes.py](api/routes.py), [api/images.py](api/images.py), [api/upload_route.py](api/upload_route.py), [api/annotations.py](api/annotations.py) | Bounded multipart uploads, shared image decoding, gallery enrollment and review, thumbnails, pipeline selection, analysis, annotation and status. |
| [face_engine/configuration.py](face_engine/configuration.py) | Pipeline registry, lazy bundle construction, initial fallback and restoration of the selected pipeline's gallery. |
| [face_engine/pipeline.py](face_engine/pipeline.py) | Per-face orchestration, matching, duplicate handling and stage timings. |
| [face_engine/research_models.py](face_engine/research_models.py) | SCRFD CPU detector, AdaFace alignment and embedding, calibrated AdaFace matcher. |
| [detector.py](face_engine/detector.py), [embedder.py](face_engine/embedder.py), [match.py](face_engine/match.py) | Bundled OpenCV YuNet detector and SFace adapter and matcher. |
| [web](web/) | Persistent shell and hash navigation, dashboard, gallery, viewer, settings, safe DOM rendering and client-side exports. Evaluation modules separate import validation, assignment and scoring, rendering, and report generation. |
| [scripts/evaluation](scripts/evaluation/) and [tests](tests/) | Benchmark, calibration, verification and regression code. The production SCRFD adapter reuses the evaluator's detector factory. The benchmark runners remain separate from normal requests. |
| [models/face](models/face/) | Tracked YuNet and SFace checkpoints and notices. Restricted optional checkpoints belong in the ignored `data/private/models/` directory. |

## System Settings in the running application

![System Settings showing preferred, active and last requested pipelines and CPU model status](docs/screenshots/system-settings.png)

The cropped drawer records one run with SCRFD + AdaFace active and ready on CPU. Bundled-model fallback and pipeline switching were verified separately. [Release verification](docs/verification/RELEASE.md)

## Architecture Trade-offs

| What we chose | What we gave up | Why |
|---|---|---|
| Local FastAPI and CPU inference | A hosted GPU service and its lower potential inference latency | The demo must run offline on one laptop. |
| Bundled YuNet + SFace with an optional preferred research bundle | A single-model installation with uniform behavior across machines | The bundled path starts without restricted checkpoints, while users can supply the preferred checkpoints subject to their applicable terms. |
| Process-local gallery and one worker | Durable enrollment and shared state across workers | Local vector reuse keeps the deployment small. |
| Manual review and optional Smart selection | Fully automatic enrollment in ambiguous cases | A person's enrollment intent can differ from the highest-scoring or largest detected face. |
| Independent browser evaluation of frozen inference | Annotation-guided changes to predictions during scoring | Correctness must be assessed against separately supplied evidence. |

## Architecture decision records

The table records implemented choices and their engineering consequences. Design rationales are retrospective unless the [collaboration record](AI_COLLAB.md) identifies a documented owner decision.

| Decision | Rationale and trade-off |
|---|---|
| Local FastAPI plus modular browser JavaScript | Keeps setup and debugging accessible and supports a dedicated photo viewer. A single-process application sets the boundary for shared state. |
| CPU inference with ONNX models | Supports local operation without a GPU service. OpenCV wraps the bundled models and ONNX Runtime supports AdaFace. The preferred bundle costs more CPU time. |
| Preferred research bundle plus fallback | Preserves the measured research configuration and independently runnable evaluator path. Failed startup or switching remains observable. |
| Pipeline-specific process galleries | Recognition vectors depend on the model and dimensions. Caching avoids re-encoding references but requires rebuilding after worker restart and consistent routing to one worker. |
| Adaptive SCRFD for gallery only | Recovers a specific missed-reference failure mode without changing group detection. A failed primary reference pass incurs the extra CPU cost. |
| Human review and optional Smart selection | The user can establish enrollment intent. Smart saves clicks in some cohorts and can choose the wrong target. |
| Immutable item IDs plus gallery revisions | Queue position can change without changing identity. Independent embedding revisions support stale request rejection and result invalidation. |
| Evaluation operates after inference | Ground truth never enters gallery enrollment, recognition, or duplicate resolution. Browser-only evaluation can be exported or recreated from independent annotations. |

## Model initialization and fallback

The app defaults to `scrfd_adaface` unless `FACEFIND_DEFAULT_PIPELINE` requests another registered pipeline. An invalid name normalizes to the preferred default and produces a note. A model initialization error triggers the alternate pipeline. If both fail, the server remains available for diagnostics. Metadata distinguishes initialization state, requested and active pipeline, initialized bundles and the fallback reason. A bundle can be selectable without being initialized. A failed switch leaves the current bundle and gallery intact because bundle construction precedes activation.

~~~mermaid
flowchart TD
    C[Read configured or preferred pipeline] --> P[Construct requested bundle]
    P -->|Ready| A[Activate bundle and expose status]
    P -->|Initialization failed| F[Construct alternate bundle]
    F -->|Ready| A
    F -->|Initialization failed| U[Serve diagnostics with engine unavailable]
~~~

## Enrollment

1. The browser uploads selected files as multipart data. The server derives identity names from file stems, validates decoded format, size and name, handles duplicates and decodes each image.
2. SCRFD uses its standard 1,280-pixel long-edge cap first. If that gallery pass finds no valid boxes, it retries at 640 pixels with the same score and NMS thresholds. Both passes preserve aspect ratio and stride padding. Boxes and landmarks are mapped back to the working image. YuNet enrollment and group-photo detection continue to use one pass.
3. A single detected face is aligned and embedded on the working image. The normalized vector and a bounded face thumbnail are stored in the active pipeline's process gallery. Zero faces after the applicable passes or a processing failure yields a rejection reason.
4. With multiple detections, the default smart policy enrolls the largest box when its area is at least five times the next-largest, its detector score is at least 0.75, and it has the uniquely highest score. Other multi-face references enter the pending queue. The explicit always-review policy queues all multi-face references. The UI opens the first pending reference automatically. Every pending card is a button for direct access, with a numbered preview, face crops, raw detector scores and confirm/skip controls. Confirm or skip advances the queue, and counters update from the response. Candidate indices follow detector order.
5. A replacement with accepted identities becomes that pipeline's gallery. An all-rejected replacement retains an existing working gallery. A replacement containing only ambiguous valid references can add to the pending queue without evicting the working embeddings. Clearing the gallery removes that pipeline's gallery and pending cache.

The upload summary includes aggregate retry attempts, recovered references and retry milliseconds. SCRFD per-reference records include the detection pass, primary time and retry time when run. A recovered reference can still enter manual review or fail embedding.

An automatic selection records the original choice, area ratio and score in the upload summary and gallery status. The in-memory gallery retains the working image and candidates for optional review under the shared 256 MiB review storage budget. If that budget is full, enrollment reports a storage error for references with multiple candidates. Correcting a choice replaces the embedding and thumbnail in place and advances the gallery revision. The original selection evidence stays available alongside the current choice. The group-photo inference path processes its own detections independently.

Gallery revisions are opaque identifiers, not filesystem paths. Thumbnail requests require the active revision and a valid index and return JPEG with `Cache-Control: no-store`. The POST lock serializes engine-affecting operations within one process. Stale preconditions return 409 before the mutation.

~~~mermaid
sequenceDiagram
    participant U as Browser
    participant A as FastAPI
    participant P as Active pipeline
    U->>A: Multipart reference files and policy
    A->>P: Detect candidates, retry empty SCRFD gallery result
    P-->>A: Boxes, landmarks and scores
    alt Single face or qualifying Smart choice
        A->>P: Align and embed intended candidate
        P-->>A: Vector
        A-->>U: Enrollment and optional review evidence
    else Ambiguous candidates
        A-->>U: Pending item with immutable ID
        U->>A: Preview by ID, then confirm or skip
        A-->>U: Updated enrollment and pending counts
    end
~~~

## Group-photo inference

1. Decode the uploaded photo, verify an active compatible gallery, then detect boxes and landmarks. YuNet scales large inputs and restores source coordinates. SCRFD scales, pads the input to stride 32, and remaps predictions.
2. Align each face using five landmarks and generate an L2-normalized vector. SFace uses OpenCV alignment and a 128-dimensional model. AdaFace uses a 112 × 112 five-point similarity transform, BGR normalization and a 512-dimensional ONNX model.
3. Compare against cached vectors using cosine similarity and pipeline-specific threshold and margin rules. AdaFace uses recognized or unknown outcomes. SFace can return possible matches. Unsuccessful embeddings are errors, and missed detections yield no row.
4. Retain the strongest duplicate assignment, with detection order breaking ties. Return working-image boxes, result fields, detected and status counts, optional candidates and conflicts, dimensions, pipeline and gallery revision, stage timings, and the annotated JPEG as a data URL. Detection, alignment, embedding, matching, annotation and backend totals are recorded when available. The browser separately measures request latency.

~~~mermaid
sequenceDiagram
    participant U as Analyze workspace
    participant A as FastAPI
    participant P as CPU pipeline
    U->>A: Preview original file
    A-->>U: Oriented bounded PNG
    U->>A: Analyze original file with pipeline and gallery preconditions
    A->>P: Decode, detect, align, embed and match
    P-->>A: Conflict-resolved predictions and stage durations
    A-->>U: Provenance, predictions and annotated JPEG
    Note over U: Evaluate results is optional after inference
~~~

## State management and concurrency

`web/app.js` owns workspace, health, pipeline, gallery revision, input image, current result and operation state. `web/api.js` handles same-origin requests and sends the active pipeline and gallery revision as preconditions on mutations. Workspace modules render the state. Navigation preserves valid results. A new photo, a changed pipeline or gallery revision, or a status refresh that can no longer validate the result invalidates it. The browser asks the server for a bounded lossless preview, then submits the original file for analysis. Export utilities download the returned JPEG and generate escaped CSV locally.

Model bundles and galleries live in one process. Mutations currently hold one process lock. Review IDs identify pending or automatic entries regardless of queue position. A stale review can be refreshed from current gallery status. Confirming an enrollment or changing an automatic selection advances the gallery revision. A skip changes the queue without changing embeddings or invalidating analysis. A no-op automatic correction also retains the revision.

~~~mermaid
flowchart LR
    G[Active gallery revision] -->|Confirm or correct a selection| N[New embedding revision]
    G -->|Successful replacement| N
    G -->|Skip pending item| G
    G -->|Failed replacement or failed model switch| G
    N --> I[Invalidate previous browser result and evaluation]
    G -->|Switch to another bundle| C[Restore that bundle's gallery or empty state]
    C --> I
~~~

~~~mermaid
sequenceDiagram
    participant X as Browser tab A
    participant Y as Browser tab B
    participant S as One FastAPI worker
    X->>S: Confirm immutable ID
    S-->>X: Enrollment and new revision
    Y->>S: Confirm stale item or stale preconditions
    S-->>Y: HTTP 409, no mutation
    Y->>S: Refresh gallery status
    S-->>Y: Current queue and revision
~~~

## Evaluation Studio

The Analyze endpoint includes additive provenance fields for the completed result. These are a unique analysis ID, a streamed SHA-256 of the uploaded image, source geometry including EXIF orientation, the gallery identity snapshot, and the matcher's active thresholds and margins. The pipeline still receives only the image, gallery and model configuration.

The browser freezes a copy of the completed response and stores the current photo preview separately. Ground-truth files are read and validated in the browser. They are never uploaded or sent to the detection or recognition pipeline. Imported annotations, manual labels, and reports are scoped to the analysis ID, gallery revision and pipeline. Changes to any of these invalidate evaluation. Browser reload clears unsaved work.

~~~mermaid
flowchart LR
    A[Completed analysis and identity snapshot] --> F[Frozen result]
    T[Independent manual or imported truth] --> V[Validate association and coordinates]
    V --> P[One-to-one IoU or manual correspondence]
    F --> P
    P --> M[Outcome counts and conditional and end-to-end rates]
    M --> X[Browser dashboard and report exports]
~~~

### Input schemas and coordinate handling

| Format | Fields |
|---|---|
| Canonical CSV | `image_filename` or `image_sha256`, `face_id`, `actual_identity`, optional `status`, and `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height` |
| Canonical JSON | A face array or an object with a `faces` array. Use the same fields, or use `bbox: [x, y, width, height]`. `image_id` can carry the SHA-256. |
| Local demo CSV | `group`, `face_number`, `source_identity`, `status` and the four `bbox_*` fields. The importer does not use `expected_label` to decide membership. |
| Local demo JSON | Uses the `faces` array plus `groups[].image`, `sha256`, `canvas_width`, `canvas_height`, and the expected face count. Imported composite boxes are marked approximate. |

Use basenames without directories. Exact file hashes are checked where supplied. Filename-only annotations require the evaluator to confirm the file corresponds to the analyzed photograph. Matching uses a case-insensitive basename and rejects ambiguous associations. Use `image_width` and `image_height` together if the coordinates are on a resized canvas. `coordinate_space` defaults to `oriented`, meaning the original image after EXIF orientation. Use `raw` for pre-orientation coordinates or `working` for the analysis image. The importer transforms box corners then maps the result to analysis pixels. The photo preview overlays use normalized coordinates. Ground-truth files are capped at 2 MiB and IoU assignment at 256 predictions and 256 actual faces per image. Missing or unreliable boxes require manual one-to-one correspondence. Mark a box unreliable using `box_reliable=false`.

### Matching and metrics

Imported boxes match predictions one-to-one at IoU >= 0.50. Assignment maximizes the number of valid matches, then total IoU, using deterministic prediction order and sorted annotation IDs for ties. TP counts matched detections, FP counts extra detections, and FN counts missed annotations. Detection precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2TP/(2TP+FP+FN). Complete manual reviews use the same counts from explicit correspondence without IoU.

| Metric | Numerator | Denominator |
|---|---|---|
| End-to-end known identification | Correct enrolled identity assignments | All annotated known faces |
| End-to-end unknown rejection | Unenrolled faces with no confirmed assignment | All annotated unknown faces |
| End-to-end false acceptance | Unenrolled faces assigned an enrolled identity | All annotated unknown faces |
| End-to-end known identification failure | Known faces with any unsuccessful outcome | All annotated known faces |
| Overall correct outcome | Correct known IDs plus correct unknown rejections | All annotated faces |
| Conditional known identification | Correct enrolled identity assignments | Matched known faces with valid recognition outcomes |
| Conditional unknown rejection | Correct rejections | Matched unknown faces with valid recognition outcomes |

A zero denominator is N/A. A missed face or processing failure is unsuccessful end to end and outside the conditional denominator. A Possible prediction is tentative and scored as unassigned. An accepted label on an extra detection is visible in its row and does not enter person-level false acceptance. Partial manual recognition denominators describe only verified labels and marked misses. JSON and CSV expose provenance, raw prediction status, outcome, scores, coordinates and metric numerators and denominators. The JPEG labels evaluation outcomes separately from the normal recognition image.

## API contracts

All paths below are under `/api`. Request schema errors can return 422. Operations supplied with stale `X-FaceFind-Pipeline` or `X-FaceFind-Gallery` preconditions return 409. FastAPI's structured error body normally contains `detail`.

| Method and route | Request | Main response and conditions |
|---|---|---|
| GET `/health` | None | Application and engine initialization, active pipeline ID, fallback/startup error. Can return HTTP 200 with engine unavailable. |
| GET `/status` | None | Gallery state, identities/counts, pending list, build errors and timings, thumbnail indices and revision, active pipeline, readiness, and image upload policy. |
| GET `/pipeline` | None | Available options, requested and active IDs, detector, recognizer, dimensions, device, model initialization timings and fallback information. |
| POST `/pipeline` | JSON `pipeline_id` | Selected pipeline and gallery status, initialization time. 400 for unknown ID, 503 if the known bundle fails to initialize. |
| POST `/gallery` | Repeated multipart `files`, optional `reference_selection_policy` (`smart` by default or `always_review`) | Build success, gallery status and per-upload outcomes, including pending and automatic counts. 503 when no engine is initialized. Individual rejected references normally return a successful HTTP response with outcome details. |
| GET `/gallery/pending/{pending_id}` | Immutable string ID, optional `interactive=true` | Identity, filename, count and preview. Interactive mode adds a clean photograph, dimensions, and ordered candidate crops, bounding boxes and detector scores. UI labels are one-based and selection indices remain zero-based. 409 for a stale ID, 500 if encoding fails. |
| POST `/gallery/select` | JSON `pending_id`, `face_index`, optional `expected_identity` | Updated gallery, pending list and selection timing. 400 for invalid selection, 409 for a stale ID, stale identity or duplicate, 500 if extraction fails. |
| POST `/gallery/skip` | JSON `pending_id`, optional `expected_identity` | Updated gallery and pending list. 409 for a stale ID. |
| GET `/gallery/automatic/{revision}/{reference_id}` | Gallery revision and immutable reference ID | Interactive candidate preview for an automatically enrolled reference. 409 for a stale ID or revision. |
| POST `/gallery/automatic/{reference_id}` | JSON `revision`, `face_index` | Updated gallery status. Changing the selected face updates the gallery revision. 400 for an invalid candidate, 409 for a stale ID or revision or 500 if extraction fails. |
| POST `/gallery/clear` | No payload required | Updated empty status for the active pipeline. |
| GET `/gallery/thumbnail/{revision}/{index}` | Path revision and index | Small JPEG or 404 for stale or unavailable thumbnail. |
| POST `/image/preview` | Multipart `file` | Oriented, bounded PNG data URL and preview dimensions. Uses the same content, decode and resource checks as analysis. |
| POST `/analyze` | Multipart `file` | Annotated JPEG data URL, working image dimensions, results per face, summary, timings, image hash and geometry, frozen gallery identity list, analysis ID and active matching rules. 400 for invalid image or missing gallery, 409 for incompatible gallery, 413 for oversized input, 415 for unsupported content, 500 for JPEG encoding failure. |

## Performance engineering

The gallery stores normalized vectors so each group request can reuse them. Detection resolution is bounded separately from working-image resolution. The API records detection, alignment, face embedding, identity matching and annotation stages, plus backend total. Browser request duration also includes request and response work around those stages. The local benchmark compares means on the same photo and reports both unreconciled timing remainders separately. [Application timing](RESEARCH.md#application-timing)

## Security and privacy

The service is intended for one desktop operator. It has no per-user authentication or multi-worker gallery coordination. The launcher binds to loopback. API requests require local Host and, if supplied, the matching Origin. The web interface uses local assets. Uploaded images, gallery vectors and review crops are process-local, with temporary file spooling where needed. Client-side evaluation is memory-only. Downloads remain on disk.

| Scenario | Handling | Verification | Remaining limitation |
|---|---|---|---|
| Large upload or exhausted review storage | Default 100 MiB/file, configurable 1–512 MiB. Gallery capped at 128 files and 2 GiB/request; individual photo request budget is the file cap plus 2 MiB. Part headers capped at 16 KiB. Shared candidate-review storage capped at 256 MiB. | [Parser and image tests](tests/test_image_ingestion.py), [release evidence](docs/verification/RELEASE.md) | Resources remain finite in the worker and OS. |
| Truncated multipart or corrupt image | Bounded parser spools above 256 KiB, offloads rollover and closes incomplete uploads. Image decoder validates content and bounds decoding to 48 MP, then works within 16 MP and 8,192 px edge limits. | [Image tests](tests/test_image_ingestion.py) | Optional formats depend on installed decoders. |
| Missing or invalid checkpoint | Startup fallback and diagnostics. Failed interactive switch preserves current state. | [Startup tests](tests/test_pipeline_startup.py), [public clone smoke](docs/verification/RELEASE.md) | Preferred weights require authorization and setup. |
| Stale reference or gallery mutation | Immutable ID and available preconditions are checked before mutation. Correction also checks embedding revision. | [Lifecycle tests](tests/test_gallery_api_lifecycle.py), Chrome two-session recovery | Process-wide shared state in one worker. |
| Incorrect truth association or malformed annotations | Browser checks filename or hash, IDs, bounds and membership. Manual correspondence for unavailable boxes. | [Evaluation regressions](tests/test_evaluation_studio.mjs) | Basename-only association requires evaluator verification. |
| Upload-stack advisories | Patched Pillow, Starlette and python-multipart pins; compatible FastAPI version. | [Advisory and compatibility record](docs/verification/RELEASE.md#final-hardening-verification) | Ongoing maintenance remains necessary. |
| Crossing the local boundary or retaining previews in caches | Same-origin checking, bounded private routes and `Cache-Control: no-store` on API responses. | [API boundary tests](tests/test_local_api_boundary.py) | No protection against another trusted local process. |
| Spreadsheet-formula injection | Shared CSV cell escaping and prefixing for potentially executable values. | [Frontend regressions](tests/test_desktop_frontend.mjs), [evaluation tests](tests/test_evaluation_studio.mjs) | Export is user-controlled; review downstream sharing. |
| Unsupported Unicode JPEG labels | Local font rendering with numbered fallback and full Unicode structured results. | [Annotation tests](tests/test_annotations.py) | Shaped Bengali and Hindi rendering was skipped on the tested Windows wheel. |

## Reproducibility

Use the [bundled quick start](README.md#installation-and-first-run) for the release at the head of this document. The clean Windows Python 3.11.9 clone ran normal analysis without ground truth, automatically fell back to YuNet + SFace and loaded Evaluation Studio. [Verification and limits](docs/verification/RELEASE.md)

For the optional CPU demo configuration, use the same Python 3.11 environment:

~~~powershell
python -m pip install -r requirements-local-demo.txt
python -m pip install --no-deps insightface==2.1
~~~

The [base pins](requirements.txt) include NumPy 1.26.4 and OpenCV contrib 4.10.0.84. The [optional pins](requirements-research.txt) include ONNX Runtime 1.32.0, ONNX 1.23.2, scikit-image 0.26.0, SciPy 1.17.1 and ml_dtypes 0.5.4. InsightFace declares `opencv-python`, but the documented separate installation preserves one `cv2` provider. The packaging metadata warning can remain despite working inference.

Only after obtaining authorization, place `scrfd_10g_bnkps.onnx` and `adaface_ir_18_webface4m_yakhyo.onnx` under the ignored `data/private/models/` directory. Remove the PowerShell override with `Remove-Item Env:FACEFIND_DEFAULT_PIPELINE -ErrorAction SilentlyContinue` or explicitly set it to `scrfd_adaface`; restart Uvicorn. Startup never downloads weights. [Exact checkpoint hashes and terms](models/face/MODEL_LICENSES.md)

## Extensibility and constraints

The pipeline registry and adapters isolate model construction, face geometry, embeddings and policy. Adding a model would require its own verified input contract, matching calibration, telemetry and rights review. In-memory galleries and browser evaluation avoid shared infrastructure while limiting durable state and multi-worker operation. The release deliberately keeps these constraints visible.

## Architectural Evolution

The browser and single worker suit a local interview workflow and simplify gallery ownership. Profile peak memory and background-operation latency before separating inference from gallery mutations. Measured queue contention or a need for concurrent operators would justify evaluating that change.

The in-memory, pipeline-specific gallery supports cache restoration within a process and keeps persisted biometric data out of the current workflow. Repeated enrollment cost could justify testing local persistence after defining retention, consent, deletion and checkpoint compatibility. Any persisted gallery would need explicit model and revision provenance.

The exact matcher is appropriate at the assignment scale. A larger deployment would justify profiling exact search before testing indexed retrieval. Compare decision agreement and open-set error rates as well as query latency. Keep independent browser-side evaluation and its provenance boundary in either design. [Validation plan](RESEARCH.md#future-research-and-validation)

## Archify artifact

The [editable specification](docs/architecture/facefind.architecture.json) and [generated HTML](docs/architecture/facefind.html) are the genuine Archify outputs for the published implementation. Archify 3.3.0 validated, delivered and checked the final Evaluation Studio diagram on November 24, 2026. Its automated browser gate passed. The HTML SHA-256 is `4b97dbe5bf257ead7c58ff0e45425014c3e51e94b1518a3fdf031455efe072c7`.

With the [official Archify tooling](https://raw.githubusercontent.com/tt-a1i/archify/main/llms.txt) installed, run from the project root:

~~~powershell
node $archifyCli finalize architecture docs/architecture/facefind.architecture.json docs/architecture/facefind.html --quality showcase --json
~~~

The generator receives the component and flow specification only. It receives no photographs, annotations, embeddings, weights or chat transcripts. The documentation revision preserves the published artifact byte for byte. The diagram's browser gate is separate from the FaceFind browser workflow checks.
