"""Small HTTP boundary around a preconfigured local model and policy."""

import hmac
import threading
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from .contracts import ContractError, MAX_BYTES, loads
from .llamacpp import BackendError


def create_app(engine, api_key=None):
    app = FastAPI(title="Granite Decisions", version="0.1.0", docs_url=None, redoc_url=None)
    slot = threading.BoundedSemaphore(1)

    @app.middleware("http")
    async def access(request, call_next):
        if api_key and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + api_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        # Browser origins are unnecessary for this machine API.
        if request.headers.get("origin"):
            return JSONResponse({"error": "browser_origin_not_allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": engine.backend.identity, "calibration": engine.backend.calibration}

    async def read_body(request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BYTES:
                raise ContractError("input_too_large")
        return loads(bytes(raw))

    @app.post("/v1/systemone")
    @app.post("/v1/decisions")
    async def decisions(request: Request):
        try:
            body = await read_body(request)
        except ContractError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not slot.acquire(blocking=False):
            return JSONResponse({"error": "model_busy"}, status_code=503)

        def infer():
            # Release in the worker, including on disconnect/cancellation.
            try:
                return engine.evaluate_request(body)
            finally:
                slot.release()
        try:
            return await run_in_threadpool(infer)
        except ContractError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except BackendError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

    return app
