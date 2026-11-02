"""Streamlit UI for SAM3 text-prompted image segmentation."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Streamlit adds this file's directory to sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch

from app.segmentation import (
    get_device_info,
    load_processor as _load_processor,
    render_segmentation,
    run_segmentation,
    state_to_detections,
)


@st.cache_resource(show_spinner="Loading SAM3 model (first run may download weights)...")
def load_processor(confidence_threshold: float):
    return _load_processor(confidence_threshold)


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
            "`uv run python scripts/download_weights.py`"
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

    from PIL import Image

    image = Image.open(uploaded).convert("RGB")
    st.subheader("Input")
    st.image(image, width="stretch")

    if not run:
        return

    processor = load_processor(confidence)
    with st.spinner(f'Segmenting "{prompt.strip()}"...'):
        state = run_segmentation(processor, image, prompt, confidence)

    detections = state_to_detections(state)
    st.subheader("Results")
    st.metric("Objects found", len(detections))

    if not detections:
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
