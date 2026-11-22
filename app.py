
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from face_engine.detector import FaceDetector
from face_engine.embedder import FaceEmbedder
from face_engine.match import FaceMatcher


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize recognition models once at startup.
    app.state.detector = FaceDetector()
    app.state.embedder = FaceEmbedder()
    app.state.matcher = FaceMatcher()

    # Initialize empty gallery state.
    app.state.gallery = None
    app.state.pending_references = []
    app.state.gallery_errors = []

    print("FaceFind recognition engine initialized.")

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
