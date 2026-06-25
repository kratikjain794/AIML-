import os
import cv2
import torch
import streamlit as st
import numpy as np

from models.colorization_model import ColorizationCNN
from utils.helper import lab_to_rgb, load_model

# ======================================
# Configuration
# ======================================

MODEL_PATH = "checkpoints/best_model.pth"
IMAGE_SIZE = 256

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ======================================
# Load Model
# ======================================

model = ColorizationCNN()

model = load_model(
    model,
    MODEL_PATH,
    device
)

# ======================================
# Streamlit UI
# ======================================

st.set_page_config(
    page_title="Image Colorization",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Image Colorization using Deep Learning")

st.write(
    "Upload a grayscale or black & white image and the trained model will generate a colorized image."
)

uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    original = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    lab = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2LAB
    )

    L = lab[:, :, 0]

    gray = L.copy()

    L = L.astype("float32") / 255.0

    L = torch.from_numpy(L)

    L = L.unsqueeze(0).unsqueeze(0)

    L = L.float().to(device)

    with torch.no_grad():

        predicted_ab = model(L)

    predicted_image = lab_to_rgb(
        L.cpu(),
        predicted_ab.cpu()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Original")

        st.image(original)

    with col2:

        st.subheader("Gray")

        st.image(gray)

    with col3:

        st.subheader("Colorized")

        st.image(predicted_image)

    save_path = "outputs/result.jpg"

    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            predicted_image,
            cv2.COLOR_RGB2BGR
        )
    )

    with open(save_path, "rb") as file:

        st.download_button(
            "Download Colorized Image",
            data=file,
            file_name="colorized_image.jpg",
            mime="image/jpeg"
        )