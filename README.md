# FaceFind

FaceFind is a local, CPU-based face identification application built for the Industry.AI Machine Learning Internship take-home. Enroll a folder of roughly 50 named reference photographs, then analyze a group photo. FaceFind detects faces, searches the reference gallery, marks unmatched faces as **Unknown**, and returns an annotated photo and structured results.

This is **open-set identification**. A group can contain people absent from the gallery, so the closest enrolled identity is accepted only when the pipeline's similarity and ambiguity rules are satisfied.

## Application preview

- TODO — Dashboard with a safe demonstration gallery.
- TODO — Gallery Manager with publishable reference portraits.
- TODO — Interactive reference review with publishable candidates.
- TODO — Analyze with a publishable group photo.
- TODO — System Settings with live timing and historical benchmark sections.

No publishable screenshots were available. The [release report](docs/verification/RELEASE.md#manual-browser-and-screenshot-pass) lists the capture steps.

## Installation and first run

Use **Python 3.11**. The commands below use Windows PowerShell and the Windows Python launcher. The first path uses only the bundled models. Package installation requires package-index access unless the packages are already cached.

~~~powershell
git clone https://github.com/aksarwar8/FaceFind.git
cd FaceFind
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
$env:FACEFIND_DEFAULT_PIPELINE = "yunet_sface"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
~~~

Open [FaceFind on loopback](http://127.0.0.1:8000/). If `py` is unavailable, substitute the absolute path to an installed Python 3.11 executable for `py -3.11`. If activation is blocked, call `.\.venv\Scripts\python.exe` directly. On macOS or Linux, use `python3.11 -m venv .venv`, `source .venv/bin/activate` and `export FACEFIND_DEFAULT_PIPELINE=yunet_sface` before launching.

### Optional research/demo configuration

In the same environment, install the combined requirements and InsightFace separately.

~~~powershell
python -m pip install -r requirements-local-demo.txt
python -m pip install --no-deps insightface==2.1
~~~

Obtain the checkpoints only through sources and permissions appropriate for your use. See the [model rights and source register](models/face/MODEL_LICENSES.md). Use these destinations for authorized local copies.

- `data/private/models/scrfd_10g_bnkps.onnx`
- `data/private/models/adaface_ir_18_webface4m_yakhyo.onnx`

Remove the earlier PowerShell environment override with `Remove-Item Env:FACEFIND_DEFAULT_PIPELINE -ErrorAction SilentlyContinue` and restart with the same Uvicorn command. To request the research pipeline explicitly, set the variable to `scrfd_adaface`. The two optional checkpoints are excluded from Git. Startup never downloads them.

The [base requirement pins](requirements.txt) include NumPy 1.26.4, OpenCV contrib 4.10.0.84, FastAPI 0.165.0, Starlette 1.10.0, python-multipart 0.0.34, Pillow 13.0.0, and Uvicorn 0.34.0. The [optional pins](requirements-research.txt) include ONNX Runtime 1.32.0, ONNX 1.23.2, scikit-image 0.26.0, ml_dtypes 0.5.4, and SciPy 1.17.1. The [combined file](requirements-local-demo.txt) includes both sets. InsightFace declares `opencv-python`, but FaceFind deliberately uses `opencv-contrib-python` as its sole `cv2` provider. The separate installation prevents both distributions from writing into the same import namespace. `pip check` can report the omitted package by name even when the actual `cv2` import and CPU inference work.

## What the app does

- Runs inference locally with two selectable pipelines and automatic startup fallback.
- Builds a searchable gallery, reports rejected references, and asks the operator to choose a face when a reference is ambiguous.
- Reuses pipeline-specific embeddings and resolves competing assignments of the same identity in a group.
- Provides a desktop Dashboard, Gallery Manager, Analyze workspace and System Settings. The photo viewer supports original and annotated views, zoom, and face highlighting.
- Exports the annotated JPEG and a spreadsheet-safe CSV. Results include model-stage and API timing where available.

## Architecture and recognition pipelines

The browser loads the plain HTML, CSS, and JavaScript UI from FastAPI and sends uploads and requests to the same local server. The backend runs the selected models, manages its process-local galleries, and returns a JPEG plus JSON. See [architecture and the Archify artifact](ARCHITECTURE.md).

| Pipeline | Purpose | Embedding |
|---|---|---:|
| SCRFD-10G-KPS + AdaFace IR-18 | Preferred research/demo choice with higher detection recall and combined identification results in the completed comparisons. Requires optional dependencies and locally authorized checkpoints. | 512 |
| YuNet + SFace | Bundled lightweight alternative and automatic fallback. Smaller model files and lower CPU latency in the saved comparisons. | 128 |

At startup FaceFind tries the requested pipeline and then the other pipeline if initialization fails. If neither initializes, the server still exposes health and status. System Settings distinguishes the preferred, last requested, and active configurations and allows switching. A failed switch leaves the working pipeline and gallery intact.

## Use FaceFind

1. Open **Dashboard** and confirm the backend and engine are ready.
2. In **Gallery Manager**, choose a reference folder or files, then build the gallery. Each file stem becomes its identity name. Review enrolled and rejected entries.
3. For pending references, inspect the numbered preview, select the intended face and confirm, or skip the reference.
4. In **Analyze**, select or drop a group image and click **Analyze photo**. Switch between original and annotated views, zoom, and select a result row to highlight a face. Download the JPEG or CSV from the results panel.
5. Switch models in **System Settings** when needed. A compatible in-memory gallery is restored if it exists. Otherwise, enroll references for that pipeline.

Under **System Settings → Reference enrollment**, **Smart Auto-Selection** is the default. With multiple detections, it enrolls the largest face when its bounding-box area is at least five times the next-largest area, its detection score is at least 0.75, and its score is strictly higher than every other candidate's. Other multiple-face references go to manual review. **Always Review Multiple Faces** sends every multiple-face reference to manual review. The browser keeps your preference for future uploads.

Use **Review automatic selection** on an enrolled reference to see the original reason, all numbered candidates, and the currently enrolled face. Choose a different face and save to update the gallery. Choose **Always Review Multiple Faces** when the intended person might be smaller than someone else in the photograph. See the [local validation results](RESEARCH.md#smart-reference-auto-selection).

| Result | Interpretation |
|---|---|
| Recognized | The pipeline accepted a gallery identity. |
| Unknown | Matching did not yield an accepted identity. Candidate identities, where shown, are suggestions only. |
| Possible match | The SFace fallback can flag a borderline candidate. AdaFace exposes recognized and unknown decisions only. |
| Processing error | A detected face could not be processed. It is counted separately from Unknown. |
| Cosine similarity | A raw comparison score. It is not a calibrated confidence probability. |
| Duplicate conflict | The strongest assignment is retained. AdaFace demotes the weaker assignment to Unknown. SFace may retain it as a possible match. |

A face missed by the detector has no result row. The app cannot infer how many unobserved faces were present.

### Image formats and limits

FaceFind checks image content. The hardening tests verified JPEG, PNG, WebP, BMP, TIFF, and AVIF with the Windows Pillow wheel. HEIC/HEIF requires an optional decoder and was not available in the tested environments. The UI reports the formats registered by the running instance.

The default is **100 MiB per image**. Set `FACEFIND_MAX_UPLOAD_MIB` to an integer from 1 to 512 before launching to change the limit. Gallery batches allow up to 128 files and 2 GiB per request. Images over 48 megapixels are rejected. Large supported images are oriented and proportionally reduced only as needed for a bounded working raster. The unannotated browser preview is smaller and lossless. Multi-page or animated images use a single frame.

## Historical CPU performance

The November 24, 2026 local FastAPI benchmark used 52 reference submissions and one genuine, unannotated group image, with one first request and five warm requests per pipeline. These are historical measurements, separate from the live timings in Settings. The CPU model was not recorded in the benchmark report.

| Pipeline | Final enrollments | Warm API mean | Warm backend mean |
|---|---:|---:|---:|
| SCRFD + AdaFace | 50 | 1,257.4 ms | 1,151.9 ms |
| YuNet + SFace | 52 | 552.7 ms | 462.6 ms |

The group image has no identity ground truth. See the [research record](RESEARCH.md) and [sanitized evidence index](docs/research/EVIDENCE.md) for experimental protocols and limitations.

## Offline operation and privacy

Inference and browser API traffic stay on the local machine when launched on loopback. Galleries, pending decoded references, and thumbnails are held in the server process and are lost when it restarts. Multipart uploads can use framework or OS temporary storage. The application does not create a durable photo archive or gallery database. The browser holds the current photo and result for the session and saves exports through its download mechanism. Run a single local worker and use authorized photos. The API has no application authentication or per-user gallery isolation.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Research pipeline unavailable | Check the two local checkpoint filenames and the optional imports. Read the reason in Settings or `/api/health`. The bundled fallback runs independently. |
| Interpreter or imports fail | Check `python --version` and `python -m pip --version` in the same shell. Recreate a virtual environment if its base interpreter was moved or removed. |
| Reference rejected or no face detected | Use a supported image below the size limit and a distinct filename. Try a clearer, larger reference face. |
| Multiple faces in a reference | Use the pending review or the enrolled reference's **Review automatic selection** button to verify the intended face. |
| Analyze is disabled | Verify that the backend and engine are ready, a compatible gallery is enrolled, and the group image decoded. |
| Pipeline switch fails | Read the Settings message and check that pipeline's checkpoint and package requirements. The previous working pipeline stays active. |
| Backend unavailable | Start Uvicorn on the same address and port, then use **Refresh status**. |

## Tests and limitations

~~~powershell
python -m pytest -q tests scripts/evaluation -rs
node --experimental-default-type=module --test tests/test_desktop_frontend.mjs
~~~

Run the full Python suite with the optional research dependencies installed. The suite does not download checkpoints or repeat model benchmarks. On **November 24, 2026**, Python 3.11.9 returned **95 passed, 2 skipped**. Both skips required Windows directory-symlink privileges. Node 22.17.1 returned **6 passed**. See the [release report](docs/verification/RELEASE.md) for clean-install HTTP results and the remaining browser checks.

The research checkpoints have unresolved or restricted rights. The detector results use a custom sample protocol. Synthetic composites and unannotated genuine photos do not establish group-photo identity accuracy. Final desktop visual verification is still pending.

- [Research and engineering decisions](RESEARCH.md)
- [Architecture and Archify](ARCHITECTURE.md)
- [AI collaboration record](AI_COLLAB.md)
- [Model licenses and provenance](models/face/MODEL_LICENSES.md)
- [Sanitized research evidence](docs/research/EVIDENCE.md)

### Annotation and preview privacy

JPEG labels use a locally installed font. Set `FACEFIND_ANNOTATION_FONT` to a trusted local TrueType or OpenType font to override the default search. Bengali and Hindi labels require glyph coverage and Pillow's RAQM shaping support. FaceFind uses numbered labels when the local renderer cannot display a name or fit it on the photo. The structured results and CSV retain the Unicode identity. No fonts are downloaded or distributed by FaceFind. API previews and annotated recognition responses send `Cache-Control: no-store`. Local downloads are saved where the browser or user chooses.
