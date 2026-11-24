import asyncio

from starlette.requests import Request
from starlette.responses import Response

from app import serialize_engine_operations


def test_local_api_rejects_foreign_browser_origin_and_rebinding_host():
    async def check(host, origin):
        headers = [(b"host", host)]
        if origin:
            headers.append((b"origin", origin))
        request = Request({
            "type": "http", "method": "GET", "path": "/api/status", "headers": headers,
        })

        async def downstream(_):
            return Response(status_code=204)

        return (await serialize_engine_operations(request, downstream)).status_code

    assert asyncio.run(check(b"127.0.0.1:8000", b"https://example.test")) == 403
    assert asyncio.run(check(b"example.test", b"http://example.test")) == 403
    assert asyncio.run(check(b"localhost:8000", b"http://localhost:8000")) == 204
    assert asyncio.run(check(b"127.0.0.1:8000", None)) == 204

def test_biometric_api_responses_are_not_stored():
    async def check():
        async def downstream(_):
            return Response(b"preview")

        for method, path in [
            ("GET", "/api/gallery/thumbnail/generation/example"),
            ("GET", "/api/gallery/pending/reference"),
            ("GET", "/api/gallery/automatic/generation/reference"),
            ("POST", "/api/image/preview"),
            ("POST", "/api/analyze"),
        ]:
            from asyncio import Lock
            from types import SimpleNamespace

            state = SimpleNamespace(operation_lock=Lock(), active_pipeline_id=None, gallery=None)
            request = Request({
                "type": "http", "method": method, "path": path,
                "headers": [(b"host", b"127.0.0.1:8000")],
                "app": SimpleNamespace(state=state),
            })
            result = await serialize_engine_operations(request, downstream)
            assert result.headers["cache-control"] == "no-store"

    asyncio.run(check())
