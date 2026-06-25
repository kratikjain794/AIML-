import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import ColorizationDataset
from models.colorization_model import ColorizationCNN
from utils.helper import save_model, create_folder


# ==========================
# Configuration
# ==========================

TRAIN_DIR = "dataset/train_small"

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
IMAGE_SIZE = 256 

CHECKPOINT_DIR = "checkpoints"

create_folder(CHECKPOINT_DIR)


# ==========================
# Device
# ==========================
  
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", device)


# ==========================
# Dataset
# ==========================

dataset = ColorizationDataset(
    image_dir=TRAIN_DIR,
    image_size=IMAGE_SIZE
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


# ==========================
# Model
# ==========================

model = ColorizationCNN().to(device)


# ==========================
# Loss
# ==========================

criterion = nn.MSELoss()


# ==========================
# Optimizer
# ==========================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================
# TensorBoard
# ==========================

writer = SummaryWriter("runs/colorization")


best_loss = float("inf")


# ==========================
# Training
# ==========================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    progress = tqdm(train_loader)

    for L, ab in progress:

        L = L.to(device)

        ab = ab.to(device)

        optimizer.zero_grad()

        output = model(L)

        loss = criterion(output, ab)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

        progress.set_postfix(
            Loss=loss.item()
        )

    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch+1} Loss : {epoch_loss:.6f}"
    )

    writer.add_scalar(
        "Training Loss",
        epoch_loss,
        epoch
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        save_model(
            model,
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

writer.close()

print("\nTraining Completed Successfully")