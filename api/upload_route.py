"""Bound multipart spooling before FastAPI binds upload parameters."""

from fastapi import HTTPException
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException, MultiPartParser

from .images import MAX_UPLOAD_BYTES, MIB


class BoundedMultipart(MultiPartParser):
    spool_max_size = 256 * 1024
    complete = False

    def on_end(self):
        self.complete = True

    def on_part_begin(self):
        super().on_part_begin()
        self.part_bytes = 0
        self.header_bytes = 0

    def _count_header_bytes(self, amount):
        self.header_bytes += amount
        if self.header_bytes > 16 * 1024:
            raise MultiPartException("Upload part headers exceed 16 KiB.")

    def on_header_field(self, data, start, end):
        self._count_header_bytes(end - start)
        super().on_header_field(data, start, end)

    def on_header_value(self, data, start, end):
        self._count_header_bytes(end - start)
        super().on_header_value(data, start, end)

    def on_header_end(self):
        self._count_header_bytes(4)
        super().on_header_end()

    def on_part_data(self, data, start, end):
        upload = self._current_part.file
        if upload is not None:
            self.part_bytes += end - start
            if self.part_bytes > MAX_UPLOAD_BYTES:
                # Keep the metadata so the route can report this file and continue.
                upload.size = MAX_UPLOAD_BYTES + 1
                return
        super().on_part_data(data, start, end)

    async def parse(self):
        form = await super().parse()
        if not self.complete:
            # An unfinished file has not yet been added to FormData.
            for file in self._files_to_close_on_error:
                await run_in_threadpool(file.close)
            raise MultiPartException("Incomplete multipart upload.")
        return form


class ImageUploadRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded_upload(request):
            if request.method != "POST" or self.path not in ("/api/gallery", "/api/analyze", "/api/image/preview"):
                return await handler(request)
            if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
                raise HTTPException(415, "Upload images as multipart form data.")

            is_gallery = self.path.endswith("/gallery")
            budget = 2 * 1024 * MIB if is_gallery else MAX_UPLOAD_BYTES + 2 * MIB
            limit_error = ("Gallery upload exceeds the batch size limit of 2 GiB." if is_gallery
                           else f"File exceeds {MAX_UPLOAD_BYTES // MIB} MiB")
            length = request.headers.get("content-length", "")
            if length.isascii() and length.isdecimal() and (len(length) > 12 or int(length) > budget):
                raise HTTPException(413, limit_error)

            async def bounded_stream():
                received = 0
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > budget:
                        raise HTTPException(413, limit_error)
                    yield chunk

            try:
                form = await BoundedMultipart(
                    request.headers, bounded_stream(), max_files=128 if is_gallery else 1,
                    max_fields=4, max_part_size=16 * 1024,
                ).parse()
            except MultiPartException as error:
                raise HTTPException(400, error.message) from error

            # FastAPI uses the request's cached form for its normal File binding.
            request._form = form
            try:
                return await handler(request)
            finally:
                await form.close()

        return bounded_upload
