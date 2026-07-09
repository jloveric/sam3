"""FastAPI service for SAM3 text-prompted image segmentation."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from app.segmentation import (
    get_device_info,
    image_to_base64_png,
    load_processor,
    render_segmentation,
    run_segmentation,
    state_to_detections,
)

_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor
    _processor = load_processor()
    yield


app = FastAPI(
    title="SAM3 Image Segmentation",
    description="Segment objects in images using a text prompt.",
    lifespan=lifespan,
)


class Detection(BaseModel):
    id: int
    score: float
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float


class SegmentResponse(BaseModel):
    prompt: str
    count: int
    detections: list[Detection]
    overlay_image_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu: str | None = None
    torch: str
    cuda_available: bool


def _get_processor():
    if _processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _processor


def _read_image(image: UploadFile) -> Image.Image:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")
    return Image.open(image.file).convert("RGB")


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    device_label, gpu_name, _ = get_device_info()
    return {
        "status": "ok",
        "device": device_label,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/segment", response_model=SegmentResponse)
def segment(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    confidence: float = Form(0.5),
    include_overlay: bool = Form(False),
) -> SegmentResponse:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1.")

    pil_image = _read_image(image)
    state = run_segmentation(_get_processor(), pil_image, prompt, confidence)
    detections = state_to_detections(state)

    overlay_b64 = None
    if include_overlay and detections:
        overlay_b64 = image_to_base64_png(render_segmentation(pil_image, state))

    return SegmentResponse(
        prompt=prompt,
        count=len(detections),
        detections=detections,
        overlay_image_base64=overlay_b64,
    )


@app.post("/segment/overlay")
def segment_overlay(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    confidence: float = Form(0.5),
) -> Response:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1.")

    pil_image = _read_image(image)
    state = run_segmentation(_get_processor(), pil_image, prompt, confidence)
    if not state["scores"]:
        return Response(status_code=404, content=b"No objects matched the prompt.")

    overlay = render_segmentation(pil_image, state)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
