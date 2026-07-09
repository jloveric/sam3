"""Shared SAM3 image segmentation helpers."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

MASK_COLORS = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 200, 64),
    (200, 64, 255),
    (64, 220, 220),
    (255, 128, 192),
    (160, 160, 64),
]


def load_processor(confidence_threshold: float = 0.5):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=device)
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def run_segmentation(processor, image: Image.Image, prompt: str, confidence_threshold: float) -> dict[str, Any]:
    processor.confidence_threshold = confidence_threshold
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_available(),
    ):
        state = processor.set_image(image)
        return processor.set_text_prompt(prompt=prompt.strip(), state=state)


def render_segmentation(image: Image.Image, state: dict[str, Any], alpha: float = 0.45) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.uint8)
    overlay = base.copy()

    scores = state["scores"]
    masks = state["masks"]
    boxes = state["boxes"]
    draw_source = Image.fromarray(overlay)
    draw = ImageDraw.Draw(draw_source)

    for index in range(len(scores)):
        color = MASK_COLORS[index % len(MASK_COLORS)]
        mask = masks[index].squeeze(0).cpu().numpy().astype(bool)
        for channel in range(3):
            overlay[..., channel][mask] = (
                alpha * color[channel] + (1 - alpha) * overlay[..., channel][mask]
            ).astype(np.uint8)

        draw_source = Image.fromarray(overlay)
        draw = ImageDraw.Draw(draw_source)
        x1, y1, x2, y2 = [int(v) for v in boxes[index].cpu().tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        score = scores[index].item()
        draw.text((x1 + 4, max(y1 - 18, 0)), f"#{index} {score:.2f}", fill=color)

    return draw_source


def state_to_detections(state: dict[str, Any]) -> list[dict[str, Any]]:
    detections = []
    for index in range(len(state["scores"])):
        x1, y1, x2, y2 = [round(v, 1) for v in state["boxes"][index].cpu().tolist()]
        detections.append(
            {
                "id": index,
                "score": round(state["scores"][index].item(), 4),
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
            }
        )
    return detections


def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_device_info() -> tuple[str, str | None, str | None]:
    if torch.cuda.is_available():
        return "CUDA", torch.cuda.get_device_name(0), None

    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_name = result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu_name = None

    if gpu_name:
        warning = (
            f"GPU detected ({gpu_name}) but PyTorch {torch.__version__} "
            f"(CUDA {torch.version.cuda}) cannot use it. "
            "Reinstall project deps with `uv sync` (expects torch 2.6 + cu124)."
        )
        return "CPU (CUDA unavailable)", gpu_name, warning

    return "CPU", None, "Running on CPU will be slow. A CUDA GPU is recommended."
