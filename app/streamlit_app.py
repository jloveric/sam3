"""Streamlit UI for SAM3 text-prompted image segmentation."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from sam3.model.sam3_image_processor import Sam3Processor

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


def render_segmentation(image: Image.Image, state: dict, alpha: float = 0.45) -> Image.Image:
    """Overlay masks and bounding boxes on the input image."""
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


@st.cache_resource(show_spinner="Loading SAM3 model (first run may download weights)...")
def load_processor(confidence_threshold: float) -> "Sam3Processor":
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=device)
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def run_segmentation(
    processor: "Sam3Processor",
    image: Image.Image,
    prompt: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    processor.confidence_threshold = confidence_threshold
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_available(),
    ):
        state = processor.set_image(image)
        return processor.set_text_prompt(prompt=prompt.strip(), state=state)


def get_device_info() -> tuple[str, str | None, str | None]:
    """Return (device label, gpu name if known, optional warning)."""
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


def main() -> None:
    st.set_page_config(page_title="SAM3 Segmentation", page_icon="🎯", layout="wide")
    st.title("SAM3 Image Segmentation")
    st.caption("Upload an image and describe what you want to segment.")

    with st.sidebar:
        st.header("Settings")
        confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
        st.divider()
        device_label, gpu_name, device_warning = get_device_info()
        st.write(f"**Device:** {device_label}")
        if gpu_name and torch.cuda.is_available():
            st.write(f"**GPU:** {gpu_name}")
        if device_warning:
            st.warning(device_warning)
        st.info(
            "Weights download automatically from Hugging Face on first segmentation. "
            "Uses your saved HF token (`HF_HOME/token`, `HF_TOKEN`, or a prior `hf auth login`). "
            "Request access at [facebook/sam3](https://huggingface.co/facebook/sam3) if needed. "
            "Pre-download: "
            "`uv run python -c \"from sam3.model_builder import download_ckpt_from_hf; "
            "print(download_ckpt_from_hf())\"`"
        )

    uploaded = st.file_uploader(
        "Drop an image here or browse",
        type=["jpg", "jpeg", "png", "webp"],
    )
    prompt = st.text_input(
        "What should be segmented?",
        placeholder='e.g. "dog", "red car", "person in white"',
    )

    run = st.button("Segment", type="primary", disabled=uploaded is None or not prompt.strip())

    if not uploaded:
        st.markdown(
            """
            ### How to use
            1. Drag and drop an image (or click to browse).
            2. Enter a short text prompt describing the objects to find.
            3. Click **Segment** to run inference.
            """
        )
        return

    image = Image.open(uploaded).convert("RGB")
    st.subheader("Input")
    st.image(image, width="stretch")

    if not run:
        return

    processor = load_processor(confidence)
    with st.spinner(f'Segmenting "{prompt.strip()}"...'):
        state = run_segmentation(processor, image, prompt, confidence)

    count = len(state["scores"])
    st.subheader("Results")
    st.metric("Objects found", count)

    if count == 0:
        st.warning("No objects matched that prompt at the current confidence threshold.")
        return

    result_image = render_segmentation(image, state)
    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.image(image, width="stretch")
    with right:
        st.markdown("**Segmentation**")
        st.image(result_image, width="stretch")

    detections = []
    for index in range(count):
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
    st.dataframe(detections, width="stretch", hide_index=True)

    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    st.download_button(
        "Download result",
        data=buffer.getvalue(),
        file_name="sam3_segmentation.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
