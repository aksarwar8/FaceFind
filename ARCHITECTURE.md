# FaceFind architecture

[Open the standalone Archify diagram](docs/architecture/facefind.html) or inspect its [editable specification](docs/architecture/facefind.architecture.json). The diagram shows the recommended local deployment. A desktop browser calls a single FastAPI worker on loopback. That worker runs CPU inference and returns structured results and an annotated JPEG. The browser prepares exports locally.

## Archify artifact

Archify **3.3.0** regenerated the hardened HTML on **November 24, 2026**. The official skill was discovered and installed in a temporary project using the [upstream instructions](https://raw.githubusercontent.com/tt-a1i/archify/main/llms.txt). The install's `doctor` check passed. The editable input was authored from the current working-tree source described below. It contains component and flow descriptions only. No dataset, photo, embedding, checkpoint, environment directory or conversation record was input to the tool. No committed-revision provenance is claimed for uncommitted working-tree source.

Run the following command from the FaceFind root with the Archify CLI installed.

~~~powershell
node $archifyCli finalize architecture docs/architecture/facefind.architecture.json docs/architecture/facefind.html --quality showcase --json
~~~

Here `$archifyCli` is the path to the installed skill's `bin/archify.mjs`. Validation, delivery, provenance checking of the rendered artifact, and the automated diagram browser check passed. The first browser attempt failed with a Chrome GPU process error in the sandbox. The authorized retry passed. Download the standalone HTML and open it locally to use the interactive viewer. It has not received a separate perceptual review. Automated diagram validation is separate from FaceFind UI verification.

## Components and boundaries

| Component | Responsibility |
|---|---|
| [app.py](app.py) | FastAPI lifespan, model startup, process caches, static UI delivery and serialization of local API mutations. |
| [api/routes.py](api/routes.py), [api/images.py](api/images.py), [api/upload_route.py](api/upload_route.py), [api/annotations.py](api/annotations.py) | Bounded multipart uploads, shared image decoding, gallery enrollment and review, thumbnails, pipeline selection, analysis, annotation and status. |
| [face_engine/configuration.py](face_engine/configuration.py) | Pipeline registry, lazy bundle construction, initial fallback and restoration of the selected pipeline's gallery. |
| [face_engine/pipeline.py](face_engine/pipeline.py) | Per-face orchestration, matching, duplicate handling and stage timings. |
| [face_engine/research_models.py](face_engine/research_models.py) | SCRFD CPU detector, AdaFace alignment and embedding, calibrated AdaFace matcher. |
| [detector.py](face_engine/detector.py), [embedder.py](face_engine/embedder.py), [match.py](face_engine/match.py) | Bundled OpenCV YuNet detector and SFace adapter and matcher. |
| [web](web/) | Persistent shell and hash navigation, dashboard, gallery, viewer, settings, safe DOM rendering and client-side exports. |
| [scripts/evaluation](scripts/evaluation/) and [tests](tests/) | Benchmark, calibration, verification and regression code. The production SCRFD adapter reuses the evaluator's detector factory. The benchmark runners remain separate from normal requests. |
| [models/face](models/face/) | Tracked YuNet and SFace checkpoints and notices. Restricted optional checkpoints belong in the ignored `data/private/models/` directory. |

## Browser state and startup

`web/app.js` owns workspace, health, pipeline, gallery revision, input image, current result and operation state. `web/api.js` handles same-origin requests and sends the active pipeline and gallery revision as preconditions on mutations. Workspace modules render the state. Navigation preserves valid results. A new photo, a changed pipeline or gallery revision, or a status refresh that can no longer validate the result invalidates it. The browser asks the server for a bounded lossless preview, then submits the original file for analysis. Export utilities download the returned JPEG and generate escaped CSV locally.

The app defaults to `scrfd_adaface` unless `FACEFIND_DEFAULT_PIPELINE` requests another registered pipeline. An invalid name normalizes to the preferred default and produces a note. A model initialization error triggers the alternate pipeline. If both fail, the server remains available for diagnostics. Metadata distinguishes initialization state, requested and active pipeline, initialized bundles and the fallback reason. A bundle can be selectable without being initialized. A failed switch leaves the current bundle and gallery intact because bundle construction precedes activation.

## Enrollment

1. The browser uploads selected files as multipart data. The server derives identity names from file stems, checks the supported format, size and name, handles duplicates and decodes each image.
2. SCRFD uses its standard 1,280-pixel long-edge cap first. If that gallery pass finds no valid boxes, it retries at 640 pixels with the same score and NMS thresholds. Both passes preserve aspect ratio and stride padding. Boxes and landmarks are mapped back to the working image. YuNet enrollment and group-photo detection continue to use one pass.
3. A single detected face is aligned and embedded on the working image. The normalized vector and a bounded face thumbnail are stored in the active pipeline's process gallery. Zero faces after the applicable passes or a processing failure yields a rejection reason.
4. With multiple detections, the default smart policy enrolls the largest box when its area is at least five times the next-largest, its detector score is at least 0.75, and it has the uniquely highest score. Other multi-face references enter the pending queue. The explicit always-review policy queues all multi-face references. A numbered JPEG preview lets the user select a candidate or skip a pending reference. Candidate indices follow detector order.
5. A replacement with accepted identities becomes that pipeline's gallery. An all-rejected replacement retains an existing working gallery. A replacement containing only ambiguous valid references can add to the pending queue without evicting the working embeddings. Clearing the gallery removes that pipeline's gallery and pending cache.

The upload summary includes aggregate retry attempts, recovered references and retry milliseconds. SCRFD per-reference records include the detection pass, primary time and retry time when run. A recovered reference can still enter manual review or fail embedding.

An automatic selection records the original choice, area ratio and score in the upload summary and gallery status. The in-memory gallery retains the working image and candidates for optional review under the shared 256 MiB review storage budget. If that budget is full, enrollment reports a storage error for references with multiple candidates. Correcting a choice replaces the embedding and thumbnail in place and advances the gallery revision. The original selection evidence stays available alongside the current choice. The group-photo inference path processes its own detections independently.

Gallery revisions are opaque identifiers, not filesystem paths. Thumbnail requests require the active revision and a valid index and return JPEG with `Cache-Control: no-store`. The POST lock serializes engine-affecting operations within one process. Stale preconditions return 409 before the mutation.

## Group-photo inference

1. Decode the uploaded photo, verify an active compatible gallery, then detect boxes and landmarks. YuNet scales large inputs and restores source coordinates. SCRFD scales, pads the input to stride 32, and remaps predictions.
2. Align each face using five landmarks and generate an L2-normalized vector. SFace uses OpenCV alignment and a 128-dimensional model. AdaFace uses a 112 × 112 five-point similarity transform, BGR normalization and a 512-dimensional ONNX model.
3. Compare against cached vectors using cosine similarity and pipeline-specific threshold and margin rules. AdaFace uses recognized or unknown outcomes. SFace can return possible matches. Unsuccessful embeddings are errors, and missed detections yield no row.
4. Retain the strongest duplicate assignment, with detection order breaking ties. Return original-frame boxes, result fields, detected and status counts, optional candidates and conflicts, dimensions, pipeline and gallery revision, stage timings, and the annotated JPEG as a data URL. Detection, alignment, embedding, matching, annotation and backend totals are recorded when available. The browser separately measures request latency.

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
| POST `/analyze` | Multipart `file` | Annotated JPEG data URL, working image dimensions, results per face, summary, timings and pipeline/gallery provenance. 400 for invalid image or missing gallery, 409 for incompatible gallery, 413 for oversized input, 415 for unsupported content, 500 for JPEG encoding failure. |

## Privacy and trade-offs

The launch command binds to loopback. API requests require a loopback Host and, when present, a matching Origin. The application has no per-user authentication and is intended for one workstation. The gallery, thumbnails, decoded review images and model state live in process memory. Replacing or clearing a gallery releases its review images. Multipart parsing may spool uploads to temporary storage. The parser enforces per-file, aggregate, file-count and header limits and cleans up incomplete uploads. Downloaded results remain wherever the browser saves them. The frontend uses local assets. The app serves its schema at `/openapi.json` without hosted Swagger or ReDoc assets. See the [rights register](models/face/MODEL_LICENSES.md) before handling weights or data.

FastAPI and vanilla JavaScript keep deployment and browser debugging small while retaining a dedicated photo viewer. A process-local gallery avoids a database and repeated embedding work, at the cost of losing enrollment on restart and preventing consistent multi-worker galleries. CPU inference avoids a remote service but adds local latency on the advanced path. Conservative acceptance reduces some false assignments while increasing abstentions. The thresholds remain workload-specific.

### Supplementary request flow

~~~mermaid
flowchart LR
    B[Browser] -->|Local upload or control request| A[FastAPI]
    A -->|Enroll or restore| G[Pipeline-specific gallery in memory]
    A -->|Decode and run active bundle| P[Detect, align and embed]
    G -->|Cached references| M[Match and resolve duplicates]
    P --> M
    M --> J[Per-face results and annotated JPEG]
    J --> B
~~~

This sequence is a supplementary diagram. The standalone [Archify export](docs/architecture/facefind.html) is the generated architecture artifact.

### Review identity and JPEG rendering

Each pending or automatically enrolled review item has an opaque immutable ID in its model-specific gallery cache. Queue position can change without altering this ID. Pending selection and skip resolve the ID in the active cache and return HTTP 409 if it is stale. Automatic correction also checks the embedding revision. Skipping an item does not change the embedding revision or invalidate existing analysis. Selecting a pending face or correcting the selected face changes the embedding revision.

The annotation renderer retains detector box coordinates and uses Pillow to draw locally shaped labels. Unsupported or oversized names fall back to face numbers backed by the structured results. API image previews, thumbnails, review crops and analyzed JPEG responses carry `Cache-Control: no-store`.

## Evaluation Studio

The Analyze endpoint includes additive provenance fields for the completed result. These are a unique analysis ID, a streamed SHA-256 of the uploaded image, source geometry including EXIF orientation, the gallery identity snapshot, and the matcher's active thresholds and margins. The pipeline still receives only the image, gallery and model configuration.

The browser freezes a copy of the completed response and stores the current photo preview separately. Ground-truth files are read and validated in the browser. They are never uploaded or sent to the detection or recognition pipeline. Imported annotations, manual labels, and reports are scoped to the analysis ID, gallery revision and pipeline. Changes to any of these invalidate evaluation. Browser reload clears unsaved work.

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
