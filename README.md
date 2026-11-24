# FaceFind

**Local, open-set face identification for group photographs.** FaceFind enrolls a named reference gallery, detects faces in a group photo, and assigns a name only when the active matching rule accepts the result. Faces outside the gallery can remain **Unknown**. The desktop browser and CPU inference run on one workstation.

[Quick start](#installation-and-first-run) · [Evaluation Studio](#evaluation-studio) · [Research report](RESEARCH.md) · [System design](ARCHITECTURE.md) · [AI collaboration](AI_COLLAB.md)

## At a glance

- **Offline CPU inference** with experimentally compared detectors and recognizers. SCRFD-10G-KPS + AdaFace IR-18 is the preferred local research bundle when available. Bundled YuNet + SFace provides a smaller, independently usable fallback.
- **Human-in-the-loop enrollment** with candidate crops, click-to-review pending cards, automatic queue advancement, and optional Smart selection with manual correction.
- **Independent evaluation** through manual ground truth or CSV/JSON import. Evaluation Studio separates missed faces, false detections, rejected known faces, and incorrect assignments from successful outcomes.
- **Measured demonstration** with a 60-identity gallery and five synthetic composites. SCRFD + AdaFace delivered **56/57 correct outcomes**, including **34/35** known identifications and **22/22** unknown rejections. The projected boxes are approximate, and the cohort does not establish general real-world accuracy. [Protocol and per-group results](RESEARCH.md#evaluation-studio-demonstration)
- **Verified release** with 103 passing Python tests, four documented Python skips, and 14 passing JavaScript tests. A fresh public clone ran the bundled models and the normal recognition workflow in Chrome. [Release verification](docs/verification/RELEASE.md)

The application release is [`9734b203f524670b695e85b0ea2088f4c6794517`](https://github.com/aksarwar8/FaceFind/commit/9734b203f524670b695e85b0ea2088f4c6794517). Any later documentation revisions refer to that frozen implementation unless explicitly stated.

## Application preview

| Workspace | What you can inspect |
|---|---|
| Dashboard | Engine status, gallery readiness, and the latest analysis and evaluation |
| Gallery | Reference enrollment, candidate crops and scores, and automatic selection corrections |
| Analyze | Original and annotated group photo, face-by-face outcomes, and recognition exports |
| Evaluation | Verified ground truth, matched faces and errors, outcome matrix, metrics, and separate exports |

[Open the interactive architecture diagram](docs/architecture/facefind.html) for a shareable view of how those workspaces interact. Browser captures from the verified demonstration contain private images and stay outside this repository.

## Installation and first run

Install **Python 3.11**. This Windows PowerShell path uses the two bundled checkpoints and works without optional packages or model downloads. Internet access can be required to install packages the first time.

~~~powershell
git clone https://github.com/aksarwar8/FaceFind.git
cd FaceFind
git checkout 9734b203f524670b695e85b0ea2088f4c6794517
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:FACEFIND_DEFAULT_PIPELINE = "yunet_sface"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
~~~

Open [FaceFind on loopback](http://127.0.0.1:8000/). If the Windows launcher is unavailable, use the absolute path to Python 3.11 in the environment-creation command. If activation is blocked, invoke `.\.venv\Scripts\python.exe` directly. On macOS or Linux, use `python3.11 -m venv .venv`, `source .venv/bin/activate` and `export FACEFIND_DEFAULT_PIPELINE=yunet_sface`. Release verification is on Windows.

## Application workflow

1. In **Gallery**, select a folder or individual reference images. Put the intended identity in the filename stem. Aim for one clearly visible intended face per reference.
2. Build the gallery. **Always Review Multiple Faces** queues every ambiguous reference. **Smart Auto-Selection** can pick a dominant face using detector evidence. Check the queued items and inspect any automatic choices you want to verify or correct.
3. In **Analyze**, choose a group photo and select **Analyze photo**. Inspect the numbered faces, assignments, Unknown outcomes, and processing errors. Download the recognition JPEG or CSV if needed.
4. For a scored review, choose **Evaluate results** after analysis. Start manual review or supply independent annotations.

## Evaluation Studio

The ordinary Analyze workflow needs no ground truth. Evaluation uses a frozen copy of the response plus the image hash, gallery revision, pipeline, and matching settings. Changing the analysis or its gallery invalidates the evaluation.

Use manual review to label each detection and add missed faces by box, or import CSV/JSON annotations. Valid imported boxes use deterministic one-to-one matching at IoU ≥ 0.50. The outcome matrix and linked photo/list filters expose errors alongside successes. Conditional scores use matched, successfully processed faces. End-to-end scores include all annotated faces in the relevant class. Numerators and denominators are visible, and incomplete manual reviews are labeled **Verified subset**. The studio supports separate JSON, CSV and JPEG exports. [Input formats and metric definitions](ARCHITECTURE.md#evaluation-studio)

## Recognition pipelines

| Bundle | Status | Face embedding |
|---|---|---:|
| YuNet + SFace | Bundled, lightweight, CPU, automatic fallback | 128 |
| SCRFD-10G-KPS + AdaFace IR-18 | Preferred for local research when the optional runtime and authorized checkpoints are available | 512 |

Switch pipelines in **Settings**. Galleries are cached separately because the two embedding spaces differ. SCRFD reference enrollment retries at 640 pixels after an empty 1,280-pixel pass. Group-photo inference uses the normal single pass. [Model setup and startup behavior](ARCHITECTURE.md#reproducibility) · [Checkpoint rights](models/face/MODEL_LICENSES.md)

### Optional preferred-pipeline setup

The bundled quick start is sufficient for Task A. To enable SCRFD + AdaFace, install the combined optional runtime and place authorized checkpoints locally, then remove the bundled-pipeline environment override before restarting. [Exact optional setup, paths and pins](ARCHITECTURE.md#reproducibility) · [Model rights](models/face/MODEL_LICENSES.md)

## Research highlights

| Evidence cohort | Result |
|---|---|
| Same 500 WIDER FACE images, custom IoU ≥ 0.50 | SCRFD improved F1 from **.6716 to .7570** versus YuNet, with mean detector latency increasing from about **100 to 526 ms**. |
| Shared recognition-only LFW cohort, 578 evaluated known probes | AdaFace and ArcFace each correctly accepted **576/578** known probes. ArcFace had fewer false unknown assignments on that cohort, while AdaFace used less CPU time per embedding. |
| Same 17 local references investigated after user testing | The gallery-only SCRFD retry raised face-detection coverage from **14/17 to 17/17** at the same detector score threshold. |
| One genuine group photo, five warm loopback requests per pipeline | Mean API latency was **0.55 s** with the bundled path and **1.26 s** with the preferred path. The photo had no identity annotations. |

The [research report](RESEARCH.md) separates these cohorts, explains why the original open-set split needed correction, and records the failures observed with dominant-face enrollment.

## Reproducibility and boundaries

FastAPI serves the local browser frontend and owns the in-memory galleries and model bundles. Inference runs in the server process. Evaluation and report preparation run in the browser. [System design and Archify provenance](ARCHITECTURE.md)

~~~powershell
python -m pytest -q tests scripts/evaluation -rs
node --experimental-default-type=module --test tests/*.mjs
~~~

The complete Python suite uses the optional runtime. See [reproducibility](ARCHITECTURE.md#reproducibility) for its installation and [verification](docs/verification/RELEASE.md) for the tested setup and skip reasons.

The application is designed for one local workstation. Gallery and evaluation state reset on restart or page reload, respectively. Large uploads can spool to the operating system's temporary storage. Exported files remain on disk. Treat face images and annotations as sensitive. Optional model rights are restricted or unresolved, and deployment rights for models and datasets require separate review. [Privacy design](ARCHITECTURE.md#security-and-privacy) · [Model rights](models/face/MODEL_LICENSES.md)

## Documentation map

| Document | Use it to find |
|---|---|
| [Research](RESEARCH.md) | Cohorts, scientific reasoning, comparative figures and negative results |
| [Architecture](ARCHITECTURE.md) | State, inference and evaluation, API contracts, security, and Archify provenance |
| [AI collaboration](AI_COLLAB.md) | Human decisions, AI contributions, original excerpts, and review outcomes |
| [Evidence index](docs/research/EVIDENCE.md) | Run provenance and sanitized aggregates |
| [Compliance matrix](docs/verification/COMPLIANCE.md) | Verified assignment scope and outstanding evidence |
