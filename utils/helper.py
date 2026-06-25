import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def lab_to_rgb(L, ab):

    if torch.is_tensor(L):
        L = L.detach().cpu().numpy()

    if torch.is_tensor(ab):
        ab = ab.detach().cpu().numpy()

    if len(L.shape) == 4:
        L = L[0]

    if len(ab.shape) == 4:
        ab = ab[0]

    L = L.squeeze()

    ab = np.transpose(ab, (1, 2, 0))

    L = L * 255.0

    ab = (ab * 128.0) + 128.0

    lab = np.zeros((256, 256, 3), dtype=np.float32)

    lab[:, :, 0] = L

    lab[:, :, 1:] = ab

    rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    return rgb


def rgb_to_lab(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256, 256))

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    lab = lab.astype(np.float32)

    L = lab[:, :, 0]

    ab = lab[:, :, 1:]

    L = L / 255.0

    ab = (ab - 128.0) / 128.0

    L = torch.from_numpy(L).unsqueeze(0).unsqueeze(0)

    ab = torch.from_numpy(ab).permute(2, 0, 1).unsqueeze(0)

    return L.float(), ab.float()


def save_model(model, path):

    torch.save(model.state_dict(), path)

    print("Model Saved Successfully")


def load_model(model, path, device):

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    model.to(device)

    model.eval()

    print("Model Loaded Successfully")

    return model


def save_image(image, save_path):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image)

    print("Image Saved :", save_path)


def show_image(image, title="Image"):

    plt.figure(figsize=(6,6))

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.show()


def show_results(gray, predicted, original=None):

    plt.figure(figsize=(15,5))

    if original is not None:

        plt.subplot(1,3,1)

        plt.imshow(original)

        plt.title("Original")

        plt.axis("off")


        plt.subplot(1,3,2)

        plt.imshow(gray, cmap="gray")

        plt.title("Gray")

        plt.axis("off")


        plt.subplot(1,3,3)

        plt.imshow(predicted)

        plt.title("Predicted")

        plt.axis("off")

    else:

        plt.subplot(1,2,1)

        plt.imshow(gray, cmap="gray")

        plt.title("Gray")

        plt.axis("off")


        plt.subplot(1,2,2)

        plt.imshow(predicted)

        plt.title("Colorized")

        plt.axis("off")

    plt.show()


def create_folder(folder_name):

    if not os.path.exists(folder_name):

        os.makedirs(folder_name)

        print(folder_name, "Created")


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )