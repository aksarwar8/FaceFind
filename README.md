# FaceFind

FaceFind is an offline, local group-photo face recognition demo. The FastAPI
service and browser UI run on the same computer; recognition requests use local
`/api` endpoints and do not call a hosted service.

## Recognition pipelines

- **Default assignment demo:** SCRFD-10G-KPS + AdaFace IR-18. This is a local
  research configuration only. SCRFD weights are restricted to non-commercial
  research, and the AdaFace checkpoint's redistribution/use rights remain
  unresolved.
- **Fallback:** YuNet + SFace, selectable from the pipeline menu. The existing
  thresholds and behavior for this pipeline are retained.

Gallery embeddings are held in memory and cached separately for each pipeline.
Switching to a pipeline with no gallery requires uploading its references.
Multiple-face references are shown for manual selection; empty or failed
replacement uploads do not remove a working gallery. Reference images are not
persisted by the service.

## Local setup

Use Python 3.11. The existing YuNet and SFace ONNX files and their license
notices are in `models/face/`. The SCRFD and AdaFace weights are deliberately
not part of this repository. For the default demo, place locally obtained,
appropriately authorized weights at:

- `data/private/models/scrfd_10g_bnkps.onnx`
- `data/private/models/adaface_ir_18_webface4m_yakhyo.onnx`

The whole `data/private/` directory is ignored by Git. The SCRFD/AdaFace path
also needs the optional packages in `requirements-research.txt`; the core app
dependencies are in `requirements.txt`. Dependency installation may require
network access, but inference itself is local. Do not treat a fresh clone as
ready for the SCRFD/AdaFace default until both local checkpoint files and the
optional runtime dependencies are supplied.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-research.txt
python -m pip install --no-deps insightface==2.1
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Install InsightFace without dependencies so it does not add the non-contrib
`opencv-python` package beside `opencv-contrib-python`; the required runtime
dependencies are listed in `requirements-research.txt`.

Open `http://127.0.0.1:8000`. Keep the service bound to loopback for local
use. If the research weights or optional runtime are unavailable, start the
tracked YuNet/SFace fallback instead:

```powershell
$env:FACEFIND_DEFAULT_PIPELINE = "yunet_sface"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The app does not download weights at startup. Use the pipeline selector in the
UI to switch between initialized configurations. Galleries are model-specific;
upload references again when a pipeline has no cached gallery.

## Evaluation and limitations

The local app timing experiment is recorded under the ignored
`data/private/evaluation/final_app_performance_20261124_01/` directory. It used
a genuine local group photograph without identity annotations, so its detected
and identified counts are operational observations, not identity accuracy.
The separate WIDER FACE and LFW research results and licensing notes are
described in [RESEARCH.md](RESEARCH.md). Custom IoU metrics are not official
WIDER FACE Easy/Medium/Hard AP, and synthetic group-photo results are not
real-world group-photo accuracy.

Run the repository's unit tests with:

```powershell
python -m unittest discover -s tests -v
```

All datasets, private photographs, research checkpoints, caches, and reports
under `data/` are excluded from Git. See [ARCHITECTURE.md](ARCHITECTURE.md) for
the app flow and model-specific gallery behavior.
