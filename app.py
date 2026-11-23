
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from face_engine.configuration import PIPELINE_LABELS, activate_pipeline


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The local assignment demo defaults to the experimentally selected model.
    # The existing YuNet/SFace fallback is initialized lazily when selected.
    app.state.pipeline_bundles = {}
    app.state.gallery_cache = {}
    app.state.pending_cache = {}
    app.state.active_pipeline_id = os.environ.get("FACEFIND_DEFAULT_PIPELINE", "scrfd_adaface")
    if app.state.active_pipeline_id not in PIPELINE_LABELS:
        app.state.active_pipeline_id = "scrfd_adaface"
    activate_pipeline(app, app.state.active_pipeline_id)
    app.state.gallery = None
    app.state.pending_references = []
    app.state.gallery_errors = []
    app.state.gallery_build_errors = []
    app.state.gallery_timings = {}
    print(f"FaceFind offline recognition engine initialized: {app.state.active_pipeline_id}.")

    yield

    print("FaceFind application shutting down.")
app = FastAPI(
    title="FaceFind",
    description="Offline group photo face recognition system",
    version="1.0.0",
    lifespan=lifespan
)


app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR),
    name="static"
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")
