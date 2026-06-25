import torch
import os


def save_checkpoint(model, optimizer, epoch, best_loss, path):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint Saved : {path}")


def load_checkpoint(model, optimizer, path, device):

    if not os.path.exists(path):

        print("No checkpoint found.")

        return model, optimizer, 0, float("inf")

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_loss = checkpoint["best_loss"]

    print(
        f"Checkpoint Loaded Successfully (Epoch {start_epoch})"
    )

    return (
        model,
        optimizer,
        start_epoch,
        best_loss
    )