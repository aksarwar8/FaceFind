
from contextlib import asynccontextmanager
from asyncio import Lock
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from face_engine.configuration import DEFAULT_PIPELINE_ID, initialize_pipeline


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.operation_lock = Lock()
    app.state.pipeline_bundles = {}
    app.state.gallery_cache = {}
    app.state.pending_cache = {}
    app.state.active_pipeline_id = None
    app.state.gallery = None
    app.state.pending_references = []
    app.state.gallery_errors = []
    app.state.gallery_build_errors = []
    app.state.gallery_timings = {}
    app.state.startup_error = None
    app.state.fallback_reason = None
    requested_pipeline = os.environ.get("FACEFIND_DEFAULT_PIPELINE", DEFAULT_PIPELINE_ID)
    initialize_pipeline(app, requested_pipeline)
    if app.state.active_pipeline_id:
        print(f"FaceFind offline recognition engine initialized: {app.state.active_pipeline_id}.")
        if app.state.fallback_reason:
            print(f"Pipeline fallback: {app.state.fallback_reason}")
    else:
        print(f"FaceFind recognition engine unavailable: {app.state.startup_error}")

    yield

    print("FaceFind application shutting down.")
app = FastAPI(
    title="FaceFind",
    description="Offline group photo face recognition system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR),
    name="static"
)

app.include_router(router)


@app.middleware("http")
async def serialize_engine_operations(request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    # Prevent browser sites from posting forms to the unauthenticated local API.
    origin = request.headers.get("origin")
    if request.url.hostname not in {"127.0.0.1", "localhost", "::1"} or (
        origin is not None and origin != f"{request.url.scheme}://{request.url.netloc}"
    ):
        return JSONResponse({"detail": "FaceFind accepts requests from its local origin."}, status_code=403)
    if request.method != "POST":
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
    async with request.app.state.operation_lock:
        state = request.app.state
        expected_pipeline = request.headers.get("X-FaceFind-Pipeline")
        expected_gallery = request.headers.get("X-FaceFind-Gallery")
        conflicting_pipeline = expected_pipeline is not None and expected_pipeline != (state.active_pipeline_id or "")
        revision = (state.gallery or {}).get("revision") or ""
        conflicting_gallery = expected_gallery is not None and expected_gallery != revision
        if conflicting_pipeline or conflicting_gallery:
            return JSONResponse({"detail": "The engine or gallery changed. Refresh status and try again."}, status_code=409)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")
