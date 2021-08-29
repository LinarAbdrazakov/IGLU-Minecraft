import torch
import torch.nn as nn

# Visual Autoencoder Parts

class VisualEncoder(nn.Module):
    def __init__(self, features_dim=512):
        super(VisualEncoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),  # (3, 64, 64) -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), #             -> (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), #             -> (64, 3, 3)
            nn.ReLU(),
            nn.Flatten(),
        )

        n_flatten = 576
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x)
        return self.linear(self.cnn(x))


class VisualDecoder(nn.Module):
    def __init__(self, features_dim=512):
        super(VisualEncoder, self).__init__()

        n_flatten = 576
        self.linear = nn.Sequential(nn.Linear(features_dim, n_flatten), nn.ReLU())

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 64, ),
            nn.ReLU(),
            nn.ConvTranspose2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 64, 3, 3)
        x = self.cnn(x)
        return x 

# Target Autoencoder Parts

class TargetEncoder(nn.Module):
    def __init__(self, features_dim=512):
        super(TargetEncoder, self).__init__()

        # input (7, 9, 11, 11)

        self.cnn = nn.Sequential(
            nn.Conv3d(7, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear = nn.Sequential(nn.Linear(512, features_dim, nn.ReLU))

    def forward(self, x):
        return self.linear(self.cnn(x))


def TargetDecoder(nn.Module):
    def __init__(self, features_dim=512):
        super(TargetDecoder, self).__init__()

        self.linear = nn.Sequential(nn.Linear(features_dim, 512))

        self.cnn = nn.Sequential(
            nn.ConvTranspose3d(),
            nn.ReLU(),
            nn.ConvTranspose3d(),
            nn.ReLU(), 
            nn.ConvTranspose3d(),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 64, 1, 2, 2)
        x = self.cnn(x)
        return x
