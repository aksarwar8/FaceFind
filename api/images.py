"""Decode bounded uploads into oriented RGB arrays before model processing."""

from importlib import import_module
import math
import os
from struct import error as StructError
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError


MIB = 1024 * 1024
MAX_UPLOAD_BYTES = int(os.getenv("FACEFIND_MAX_UPLOAD_MIB", "100")) * MIB
if not MIB <= MAX_UPLOAD_BYTES <= 512 * MIB:
    raise ValueError("FACEFIND_MAX_UPLOAD_MIB must be between 1 and 512.")

MAX_SOURCE_PIXELS = 48_000_000
MAX_WORKING_PIXELS = 16_000_000
MAX_WORKING_EDGE = 8192
MAX_PENDING_BYTES = 256 * MIB
BASE_FORMATS = ("JPEG", "PNG", "WEBP", "BMP", "TIFF")
OPTIONAL_FORMATS = ("HEIF", "AVIF")

for module, register in (("pillow_heif", "register_heif_opener"), ("pillow_avif", None)):
    try:
        plugin = import_module(module)
        if register:
            getattr(plugin, register)()
    except (ImportError, OSError, AttributeError):
        # Optional codecs must not prevent the bundled formats from loading.
        continue
Image.init()
FORMATS = tuple(name for name in (*BASE_FORMATS, *OPTIONAL_FORMATS) if name in Image.OPEN)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}

ORIENTATION = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


class ImageInputError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def upload_policy():
    return {"max_bytes": MAX_UPLOAD_BYTES, "formats": list(FORMATS)}


def _optional_format(file):
    """Identify ISO BMFF photo brands even when their decoder is absent."""
    file.seek(0)
    header = file.read(64)
    if header[4:8] != b"ftyp":
        return None
    brands = [header[8:12], *(header[i:i + 4] for i in range(16, min(len(header), int.from_bytes(header[:4], "big")), 4))]
    if any(brand in (b"avif", b"avis") for brand in brands):
        return "AVIF"
    if any(brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1") for brand in brands):
        return "HEIF"
    return None


def _target(size, pixels, edge):
    width, height = size
    scale = min(1, edge / max(size), math.sqrt(pixels / (width * height)))
    return max(1, int(width * scale)), max(1, int(height * scale))


def decode_image(upload, *, pixels=MAX_WORKING_PIXELS, edge=MAX_WORKING_EDGE, geometry=None):
    """Read the spooled file directly and return a contiguous uint8 RGB array."""
    file = upload.file
    size = upload.size
    if size is None:
        size = file.seek(0, os.SEEK_END)
    if size > MAX_UPLOAD_BYTES:
        raise ImageInputError(f"File exceeds {MAX_UPLOAD_BYTES // MIB} MiB", 413)
    if not size:
        raise ImageInputError("Empty image file")
    file.seek(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(file, formats=FORMATS) as image:
                original_size, source_format = image.size, image.format
                if math.prod(image.size) > MAX_SOURCE_PIXELS:
                    raise ImageInputError("Image exceeds the 48-megapixel decode limit", 413)
                if min(image.size) < 2 or max(image.size) > 1280 * min(image.size):
                    raise ImageInputError("Image dimensions are too small or too narrow for face detection")
                original_orientation = image.getexif().get(274, 1) if geometry is not None and source_format == "TIFF" else None
                if original_orientation is not None:
                    # Pillow exposes an already oriented TIFF size; read the raw IFD dimensions.
                    original_size = (int(image.tag_v2[256]), int(image.tag_v2[257]))

                # JPEG can decode at a reduced size without allocating the full raster.
                if image.format == "JPEG":
                    image.draft("RGB", _target(image.size, pixels, edge))
                image.load()
                orientation = image.getexif().get(274, 1)
                if image.format == "HEIF":
                    orientation = image.info.get("original_orientation") or orientation
                target = _target(image.size, pixels, edge)

                # Palette and high-bit-depth modes need an RGB-compatible conversion.
                if image.mode.startswith("I;16") or image.mode == "I" and image.format == "PNG":
                    image = image.convert("I").point(lambda value: value / 257).convert("L")
                if image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    if rgba.size != target:
                        rgba = rgba.resize(target, Image.Resampling.LANCZOS)
                    image = Image.new("RGB", rgba.size, "white")
                    image.paste(rgba, mask=rgba.getchannel("A"))
                elif image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                if image.size != target:
                    image = image.resize(target, Image.Resampling.LANCZOS, reducing_gap=2.0)
                if orientation in ORIENTATION:
                    image = image.transpose(ORIENTATION[orientation])
                if geometry is not None:
                    # TIFF loaders may already have applied orientation during load.
                    source_orientation = orientation if original_orientation is None else original_orientation
                    geometry.update(original_width=original_size[0], original_height=original_size[1],
                                    orientation=source_orientation if source_orientation in ORIENTATION else 1)
                return np.array(image.convert("RGB") if image.mode != "RGB" else image)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageInputError("Image dimensions exceed safe decode limits", 413) from error
    except UnidentifiedImageError as error:
        codec = _optional_format(file)
        if codec and codec not in FORMATS:
            raise ImageInputError(f"{codec} decoder is unavailable on this server", 415) from error
        raise ImageInputError("Unsupported or unrecognized image content", 415) from error
    except ImageInputError:
        raise
    except (OSError, SyntaxError, ValueError, RuntimeError, OverflowError, EOFError, StructError) as error:
        raise ImageInputError("Image is corrupt or could not be decoded") from error
