#!/usr/bin/env python3
"""
DeepShield API server for EC2 deployment.

Usage:
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from deepshield import ProtectionConfig, protect_image
from deepshield.protect import build_runtime


APP_START_TIME = time.time()
DEFAULT_DEVICE = os.getenv("DEEPSHIELD_DEVICE", "cuda")
DEFAULT_MODEL_ID = os.getenv("DEEPSHIELD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEFAULT_EPSILON = float(os.getenv("DEEPSHIELD_EPSILON", "20"))
DEFAULT_STEPS = int(os.getenv("DEEPSHIELD_STEPS", "300"))
DEFAULT_EOT = int(os.getenv("DEEPSHIELD_N_EOT", "8"))
DEFAULT_FREQ_LAMBDA = float(os.getenv("DEEPSHIELD_FREQ_LAMBDA", "8.0"))
DEFAULT_USE_DENOISING = os.getenv("DEEPSHIELD_USE_DENOISING_LOSS", "false").lower() == "true"
DEFAULT_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("DEEPSHIELD_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
MAX_UPLOAD_MB = int(os.getenv("DEEPSHIELD_MAX_UPLOAD_MB", "15"))


class HealthResponse(BaseModel):
    status: str
    runtime_loaded: bool
    uptime_seconds: int
    device: str
    model_id: str


app = FastAPI(title="DeepShield API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runtime = None
_startup_error: Optional[str] = None


def build_default_config(
    *,
    epsilon: Optional[float] = None,
    steps: Optional[int] = None,
    n_eot: Optional[int] = None,
) -> ProtectionConfig:
    epsilon_value = (epsilon if epsilon is not None else DEFAULT_EPSILON) / 255.0
    return ProtectionConfig(
        epsilon=epsilon_value,
        step_size=epsilon_value / 100.0,
        num_steps=steps if steps is not None else DEFAULT_STEPS,
        n_eot=n_eot if n_eot is not None else DEFAULT_EOT,
        freq_lambda=DEFAULT_FREQ_LAMBDA,
        use_denoising_loss=DEFAULT_USE_DENOISING,
        model_id=DEFAULT_MODEL_ID,
        device=DEFAULT_DEVICE,
    )


@app.on_event("startup")
def load_runtime_on_startup() -> None:
    global _runtime, _startup_error

    try:
        _runtime = build_runtime(build_default_config())
        _startup_error = None
    except Exception as exc:  # pragma: no cover
        _runtime = None
        _startup_error = str(exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _runtime is not None else "starting",
        runtime_loaded=_runtime is not None,
        uptime_seconds=int(time.time() - APP_START_TIME),
        device=DEFAULT_DEVICE,
        model_id=DEFAULT_MODEL_ID,
    )


@app.get("/")
def root():
    return {
        "service": "deepshield-api",
        "status": "ok" if _runtime is not None else "starting",
        "health": "/health",
        "process": "/process",
        "startup_error": _startup_error,
    }


@app.post("/process")
async def process_image(
    image: Optional[UploadFile] = File(default=None),
    file: Optional[UploadFile] = File(default=None),
    epsilon: Optional[float] = Form(default=None),
    steps: Optional[int] = Form(default=None),
    n_eot: Optional[int] = Form(default=None),
):
    upload = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail="Missing image upload.")

    if _runtime is None:
        raise HTTPException(
            status_code=503,
            detail=_startup_error or "DeepShield runtime is not ready. Start or warm the EC2 worker and retry.",
        )

    suffix = Path(upload.filename or "upload.png").suffix or ".png"

    with tempfile.TemporaryDirectory(prefix="deepshield-api-") as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        output_path = Path(tmpdir) / "protected.png"

        size_bytes = 0
        with input_path.open("wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_MB} MB.")
                buffer.write(chunk)

        cfg = build_default_config(epsilon=epsilon, steps=steps, n_eot=n_eot)
        protected_path = Path(
            protect_image(
                str(input_path),
                str(output_path),
                cfg=cfg,
                runtime=_runtime,
                also_save_jpeg=False,
            )
        )

        download_name = f"{Path(upload.filename or 'upload').stem}_deepshield.png"
        response_path = Path(tmpdir) / download_name
        shutil.copy2(protected_path, response_path)

        return FileResponse(
            path=response_path,
            media_type="image/png",
            filename=download_name,
        )
