import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from efficientnet_pytorch import EfficientNet


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.block1 = self.conv_block(
            c_in=1, c_out=256, dropout=0.3, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(
            c_in=256, c_out=128, dropout=0.4, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(
            c_in=128, c_out=64, dropout=0.5, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=56, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*1*1, 32)
        self.fc2 = nn.Linear(32, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        x = self.maxpool(x)
        #x = self.dropout2
        #x = x.view(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block
    

class Baseline_enet(nn.Module):
    def __init__(self, out_dim):
        super(Baseline_enet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(12, 36, 3, stride=1, padding=1, bias=False)
        self.mybn1 = nn.BatchNorm2d(6)
        self.mybn2 = nn.BatchNorm2d(12)
        self.mybn3 = nn.BatchNorm2d(36)

        self.enet = timm.create_model('efficientnet_b0', pretrained=True)
        self.enet.conv_stem.weight = nn.Parameter(
            self.enet.conv_stem.weight.repeat(1, 12, 1, 1))

        self.dropout = nn.Dropout(0.5)
        self.enet.blocks[5] = nn.Identity()
        self.enet.blocks[6] = nn.Sequential(
            nn.Conv2d(
                self.enet.blocks[4][2].conv_pwl.out_channels, self.enet.conv_head.in_channels, 1),
            nn.BatchNorm2d(self.enet.conv_head.in_channels),
            nn.ReLU6(),
        )
        self.myfc = nn.Linear(self.enet.classifier.in_features, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = F.relu6(self.mybn1(self.conv1(x)))
        x = F.relu6(self.mybn2(self.conv2(x)))
        x = F.relu6(self.mybn3(self.conv3(x)))
        x = self.enet(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 1000)
        x = self.myfc(self.dropout(x))
        return x
    
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnetmodel = EfficientNet.from_pretrained('efficientnet-b3')

        self.fc = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),
                                nn.Linear(512, 1),nn.ReLU(),
                                nn.Sigmoid())
        
        self.fc1 = nn.Sequential(nn.Linear(3072,1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.resnetmodel(x)
        #x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 3072)
        
        return self.fc(x)  

if __name__ == "__main__":
    model = Model()
    #print(model)
    x = torch.randn((2, 3,512,512))
    out = model(x)
    
    print(out)
