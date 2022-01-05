import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models



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
    def __init__(self, out_dim=1):
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
        self.enet.classifier = nn.Sigmoid()

    def extract(self, x):
        x = F.relu6(self.mybn1(self.conv1(x)))
        x = F.relu6(self.mybn2(self.conv2(x)))
        x = F.relu6(self.mybn3(self.conv3(x)))
        x = self.enet(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        #x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 1000)
        x = self.myfc(self.dropout(x))
        return self.enet.classifier(x)


class Model(nn.Module):
    def __init__(self, pretrained):
        super(Model, self).__init__()
        self.pretrained = pretrained
        if self.pretrained == False:
            self.model = models.mobilenet_v3_small()
        else:
            self.model = models.mobilenet_v3_small(pretrained=True)

        self.fc = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),
                                nn.Linear(512, 1),nn.ReLU(),
                                nn.Sigmoid())

        self.fc1 = nn.Sequential(nn.Linear(3072,1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.model(x)
        #x = F.avg_pool2d(x, x.size()[2:]).reshape(-1, 3072)

        return self.fc(x)


def initialize_model(model_name, num_classes, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        #Resnet18
        model_ft = models.resnet34(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224

    elif model_name == "alexnet":
        #Alexnet
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224

    elif model_name == "vgg":
        #VGG11_bn
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        input_size = 224

    elif model_name == "squeezenet":
        #Squeezenet
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),nn.Sigmoid())
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        #Densenet
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

    elif model_name == "inception":
        #Inception v3
        model_ft = models.inception_v3(pretrained=use_pretrained)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

if __name__ == "__main__":
    model = Model(pretrained=False)
    #model = models.mobilenet_v3_small()
    print(model)
    x = torch.randn((4, 3,512,512))
    out = model(x)

    print(out)
