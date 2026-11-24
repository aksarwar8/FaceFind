import asyncio
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import MultiPartException

from api.images import ImageInputError, decode_image
from api.upload_route import BoundedMultipart


def upload(image, format, **options):
    contents = BytesIO()
    image.save(contents, format, **options)
    contents.seek(0)
    return UploadFile(contents, filename="not-the-image-format.txt")


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "BMP", "TIFF"])
def test_decode_by_content(format):
    image = decode_image(upload(Image.new("L", (24, 12), 160), format))
    assert image.shape == (12, 24, 3)
    assert image.dtype == np.uint8 and image.flags.c_contiguous
    assert np.all(image == 160)


@pytest.mark.parametrize("format", ["JPEG", "TIFF"])
def test_exif_orientation_is_applied_once(format):
    source = Image.new("RGB", (12, 6), "red")
    source.paste("blue", (6, 0, 12, 6))
    exif = source.getexif()
    exif[274] = 6
    image = decode_image(upload(source, format, exif=exif))
    assert image.shape == (12, 6, 3)
    assert image[0, 0, 0] > 200
    assert image[-1, 0, 2] > 200


def test_alpha_and_high_bit_depth_use_rgb():
    rgba = Image.new("RGBA", (3, 2), (255, 0, 0, 0))
    rgba.putpixel((1, 0), (255, 0, 0, 255))
    result = decode_image(upload(rgba, "PNG"))
    assert result[0, 0].tolist() == [255, 255, 255]
    assert result[0, 1].tolist() == [255, 0, 0]
    sixteen_bit = Image.fromarray(np.full((2, 2), 32768, dtype=np.uint16))
    assert decode_image(upload(sixteen_bit, "PNG"))[0, 0].tolist() == [127] * 3


def test_resize_and_decode_limits():
    picture = Image.new("RGB", (400, 200))
    assert decode_image(upload(picture, "PNG"), pixels=10000, edge=100).shape == (50, 100, 3)
    with patch("api.images.MAX_SOURCE_PIXELS", 1000):
        with pytest.raises(ImageInputError, match="decode limit") as error:
            decode_image(upload(picture, "PNG"))
        assert error.value.status_code == 413
    with pytest.raises(ImageInputError, match="Unsupported"):
        decode_image(UploadFile(BytesIO(b"not an image"), filename="example.jpg"))


def test_truncated_jpeg_is_rejected():
    original = upload(Image.new("RGB", (128, 128), "red"), "JPEG").file.getvalue()
    truncated = UploadFile(BytesIO(original[:-60]), filename="truncated.jpg")
    with pytest.raises(ImageInputError, match="corrupt"):
        decode_image(truncated)


def test_multipart_discards_oversized_part_and_keeps_following_file():
    boundary = b"test-boundary"
    def part(name, data):
        return (b"--" + boundary + b'\r\nContent-Disposition: form-data; name="files"; filename="'
                + name + b'"\r\n\r\n' + data + b"\r\n")

    body = part(b"oversized.png", b"x" * 50) + part(b"next.png", b"valid") + b"--" + boundary + b"--\r\n"
    async def chunks():
        for offset in range(0, len(body), 17):
            yield body[offset:offset + 17]

    async def check():
        parser = BoundedMultipart(Headers({"content-type": "multipart/form-data; boundary=test-boundary"}), chunks())
        form = await parser.parse()
        try:
            oversized, following = form.getlist("files")
            with pytest.raises(ImageInputError, match="File exceeds"):
                decode_image(oversized)
            assert len(await oversized.read()) <= 10
            assert await following.read() == b"valid"
        finally:
            await form.close()

    with patch("api.upload_route.MAX_UPLOAD_BYTES", 10), patch("api.images.MAX_UPLOAD_BYTES", 10):
        asyncio.run(check())


def test_truncated_multipart_closes_part_even_without_a_final_boundary():
    async def chunks():
        yield (b'--form\r\nContent-Disposition: form-data; name="files"; filename="partial.jpg"'
               b'\r\n\r\n' + b'x' * (300 * 1024))

    async def check():
        parser = BoundedMultipart(Headers({"content-type": "multipart/form-data; boundary=form"}), chunks())
        with pytest.raises(MultiPartException, match="Incomplete multipart upload"):
            await parser.parse()
        assert parser._files_to_close_on_error
        assert all(file.closed for file in parser._files_to_close_on_error)

    asyncio.run(check())


def test_native_avif_decode_when_available():
    from api.images import FORMATS

    if "AVIF" not in FORMATS:
        pytest.skip("Pillow AVIF decoder is unavailable")
    result = decode_image(upload(Image.new("RGB", (24, 12), (140, 140, 140)), "AVIF"))
    assert result.shape == (12, 24, 3)
    assert np.abs(result.astype(int) - 140).max() <= 3


def test_ordinary_image_spool_rollover_leaves_health_and_status_responsive(monkeypatch):
    import tempfile
    from threading import Event
    from starlette import formparsers
    from app import app

    entered = Event()
    release = Event()

    class PausedSpool(tempfile.SpooledTemporaryFile):
        def rollover(self):
            entered.set()
            if not release.wait(2):
                raise AssertionError("Upload rollover blocked peer requests on the event loop")
            return super().rollover()

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", PausedSpool)
    pixels = np.random.default_rng(44).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    photo = upload(Image.fromarray(pixels), "PNG").file.getvalue()
    prefix = b'--form\r\nContent-Disposition: form-data; name="file"; filename="ordinary.png"\r\n\r\n'
    body = prefix + photo + b'\r\n--form--\r\n'

    async def exchange(method, path, data=b""):
        chunks = iter(data[offset:offset + 65536] for offset in range(0, len(data), 65536))
        responses = []

        async def receive():
            chunk = next(chunks, b"")
            return {"type": "http.request", "body": chunk, "more_body": bool(chunk)}

        async def send(message):
            responses.append(message)

        await app({
            "type": "http", "method": method, "path": path, "scheme": "http",
            "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"multipart/form-data; boundary=form")],
            "query_string": b"", "http_version": "1.1", "server": ("127.0.0.1", 80),
        }, receive, send)
        return responses[0]["status"]

    async def check():
        async with app.router.lifespan_context(app):
            preview = asyncio.create_task(exchange("POST", "/api/image/preview", body))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                assert await asyncio.wait_for(exchange("GET", "/api/health"), 0.75) == 200
                assert await asyncio.wait_for(exchange("GET", "/api/status"), 0.75) == 200
            finally:
                release.set()
                assert await preview == 200
            # Exercise the streaming aggregate limit without allocating a 2 GiB test body.
            with patch("api.upload_route.MIB", 512):
                assert await exchange("POST", "/api/gallery", body) == 413
                assert await exchange("GET", "/api/status") == 200

    asyncio.run(check())
