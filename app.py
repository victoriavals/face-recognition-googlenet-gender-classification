"""
Gender Classification — GoogLeNet

A Streamlit frontend application to classify gender (Male / Female)
from face images using a pretrained GoogLeNet model.

Usage:
    streamlit run app.py
"""

import logging
import os
from typing import Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "best_model.pth")
IMAGE_SIZE: int = 224
LABELS: dict[int, str] = {0: "Female", 1: "Male"}
THRESHOLD: float = 0.5

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ──────────────────────────────────────────────────────────────
# Model Loading (cached to avoid reloading on each interaction)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model() -> nn.Module:
    """Load the trained GoogLeNet gender classification model.

    Returns:
        nn.Module: The model in evaluation mode.

    Raises:
        FileNotFoundError: If the model weights file is not found.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model on device: %s", device)

    model = models.googlenet(weights=None, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully.")
    return model


def get_device() -> torch.device:
    """Detect and return the available compute device.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for model inference.

    The preprocessing steps replicate exactly what the training notebook
    does: resize to 224×224, convert to float32 [0, 1], and permute
    to (1, C, H, W).  No additional ImageNet normalisation is applied
    because the training pipeline did not use it.

    Args:
        image: A PIL Image (RGB).

    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 3, 224, 224).
    """
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)
    return tensor


# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────
def predict(model: nn.Module, image: Image.Image) -> Tuple[str, float]:
    """Run gender classification on a single image.

    Args:
        model: The loaded GoogLeNet model.
        image: Input PIL Image.

    Returns:
        Tuple[str, float]: Predicted label ("Male" or "Female") and
            the confidence score in [0, 1].
    """
    device = get_device()
    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(tensor).squeeze()
        probability = torch.sigmoid(output).item()

    if probability >= THRESHOLD:
        return LABELS[1], probability
    return LABELS[0], 1.0 - probability


# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for the Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="Gender Classification — GoogLeNet",
        page_icon="🧑",
        layout="centered",
    )

    # Title
    st.title("🧑 Gender Classification")
    st.markdown(
        "Classify a face image as **Male** or **Female** using a "
        "pretrained **GoogLeNet** deep‑learning model."
    )
    st.divider()

    # Load model
    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # Sidebar — input method selector
    with st.sidebar:
        st.header("⚙️ Settings")
        input_method = st.radio(
            "Choose input method:",
            ("Upload Image", "Webcam Capture"),
            index=0,
        )
        st.divider()
        st.caption(
            "**Tech Stack:** PyTorch · GoogLeNet · Streamlit\n\n"
            "Model trained on a subset of the CelebA dataset."
        )

    # Main area
    image: Image.Image | None = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    else:  # Webcam Capture
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo)

    # Display prediction
    if image is not None:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("📷 Input Image")
            st.image(image, use_container_width=True)

        with col_result:
            st.subheader("📊 Prediction")

            with st.spinner("Classifying…"):
                label, confidence = predict(model, image)

            if label == "Male":
                st.markdown(
                    f"### 🔵 **{label}**"
                )
            else:
                st.markdown(
                    f"### 🔴 **{label}**"
                )

            st.metric(label="Confidence", value=f"{confidence:.1%}")
            st.progress(confidence)


if __name__ == "__main__":
    main()
