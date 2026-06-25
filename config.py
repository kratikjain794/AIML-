import os
import torch

# ==========================
# DATASET PATHS
# ==========================

TRAIN_DIR = "dataset/train_samples"
VALID_DIR = "dataset/valid_samples"
TEST_DIR = "dataset/test_samples"

# ==========================
# IMAGE SETTINGS
# ==========================

IMAGE_SIZE = 256

# ==========================
# TRAINING
# ==========================

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 1e-3

# ==========================
# CHECKPOINTS
# ==========================

CHECKPOINT_DIR = "checkpoints"

BEST_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

LAST_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "last_model.pth"
)

# ==========================
# OUTPUT
# ==========================

OUTPUT_DIR = "outputs"

RUNS_DIR = "runs"

# ==========================
# RESUME TRAINING
# ==========================

RESUME = True

# ==========================
# CONTINUE LEARNING
# ==========================

CONTINUE_ON_NEW_DATASET = True

# ==========================
# SAVE
# ==========================

SAVE_EVERY = 1

# ==========================
# DEVICE
# ==========================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)