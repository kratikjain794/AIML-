import os
import cv2
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ColorizationDataset(Dataset):

    def __init__(self, image_dir, image_size=256):

        self.image_dir = image_dir

        self.image_list = []

        for file in os.listdir(image_dir):

            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):

                self.image_list.append(file)

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def __len__(self):

        return len(self.image_list)


    def __getitem__(self, index):

        image_name = self.image_list[index]

        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB
        )

        lab = lab.astype(np.float32)

        L = lab[:, :, 0]

        ab = lab[:, :, 1:]

        L = L / 255.0

        ab = (ab - 128) / 128.0

        L = torch.from_numpy(L).unsqueeze(0)

        ab = torch.from_numpy(ab).permute(2, 0, 1)

        return L.float(), ab.float()