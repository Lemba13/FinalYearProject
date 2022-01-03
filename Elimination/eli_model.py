import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d

class AutoEncoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=1),
            nn.ReLU(True),
            #nn.Dropout(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 5, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64 , 5, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 5, padding = 1),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3*3*512, 256),
            nn.ReLU(True),
            nn.Linear(256, encoded_space_dim)
        )
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 3*3*512),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 128, 3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 5, stride=2, output_padding=1),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 7, stride=1),
        )


    def encforward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
    def dncforward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        #x = torch.sigmoid(x)
        return x
    
    def forward(self,x):
        x = self.encforward(x)
        x = self.dncforward(x)
        return x




if __name__ == "__main__":
    d=32
    model = AutoEncoder(encoded_space_dim=d)#, fc2_input_dim=128)
    ##decoder = Decoder(encoded_space_dim=d)#, fc2_input_dim=128)
    x = torch.randn((2, 3, 512,512))
    res = model(x)
    #res = decoder(out)

    assert x.shape == res.shape
    criterion = nn.MSELoss()
    loss = criterion(res,x)
    print(res.shape)
    print(loss)
