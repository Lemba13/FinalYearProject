import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d

class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32 , 5, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(31 * 31 * 64, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    

if __name__ == "__main__":
    d=16
    encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)
    x = torch.randn((2, 3, 512,512))
    out = encoder(x)

    print(out)
