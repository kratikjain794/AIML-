import os
import cv2
import torch

from models.colorization_model import ColorizationCNN
from utils.helper import (
    lab_to_rgb,
    load_model,
    save_image,
    show_results,
    create_folder
)


# ======================================
# Configuration
# ======================================

IMAGE_PATH = "dataset/test_samples/Places365_val_00000006.jpg"

MODEL_PATH = "checkpoints/best_model.pth"

OUTPUT_FOLDER = "outputs"

IMAGE_SIZE = 256


create_folder(OUTPUT_FOLDER)


# ======================================
# Device
# ======================================

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
# Read Image
# ======================================

image = cv2.imread(IMAGE_PATH)

original = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

original = cv2.resize(
    original,
    (IMAGE_SIZE, IMAGE_SIZE)
)


# ======================================
# Convert RGB to LAB
# ======================================

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


# ======================================
# Prediction
# ======================================

with torch.no_grad():

    predicted_ab = model(L)


predicted_image = lab_to_rgb(
    L.cpu(),
    predicted_ab.cpu()
)


# ======================================
# Save Output
# ======================================

save_path = os.path.join(
    OUTPUT_FOLDER,
    "colorized_output.jpg"
)

save_image(
    predicted_image,
    save_path
)


# ======================================
# Show Result
# ======================================

show_results(
    gray,
    predicted_image,
    original
)

print("\nDone")