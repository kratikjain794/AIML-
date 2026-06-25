import torch
import torch.nn as nn


class ColorizationCNN(nn.Module):

    def __init__(self):

        super(ColorizationCNN, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )


        self.middle = nn.Sequential(

            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True)

        )


        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),


            nn.Conv2d(
                64,
                2,
                kernel_size=1
            ),

            nn.Tanh()

        )


    def forward(self, x):

        x = self.encoder(x)

        x = self.middle(x)

        x = self.decoder(x)

        return x