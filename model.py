import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from common import Config


cfg = Config()

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        resnet = models.resnet18(num_classes=365) 
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.
                                sum(dim=1).unsqueeze(1).data) 
        if torch.cuda.is_available(): # and only if gpu is available
            resnet_gray_weights = torch.load('Pretrained/resnet_gray_weights.pth.tar')
            resnet.load_state_dict(resnet_gray_weights)
            print('Pretrained ResNet-gray weights loaded')
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    def forward(self, x):
        return self.midlevel_resnet(x)


class Upsample(nn.Module):
    def __init__(self, c=False):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(     
          nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
        ) 
        self.c = c
        self.conva = nn.Conv2d(32, cfg.bins, kernel_size=3, stride=1, padding=1)
        self.convb = nn.Conv2d(32, cfg.bins, kernel_size=3, stride=1, padding=1)
        self.convr = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.soft = nn.Softmax(dim=1)
        self.up = nn.Upsample(scale_factor=2)
    def forward(self, x):
        out = self.upsample(x)
        if self.c: 
            a = self.conva(out)
            a = self.up(a)
            b = self.convb(out)
            b = self.up(b)
            return self.soft(a), self.soft(b)
        else:
            out = self.convr(out)
            out = self.up(out)
            return out


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.down_net = Backbone()
        if cfg.is_classification:
            self.up_net = Upsample(True)
        else:
            self.up_net = Upsample()

    def forward(self, x):
        x = self.down_net(x)
        x = self.up_net(x)
        return x


if __name__ == "__main__":

    colornet = ColorNet()
    data = torch.ones(3, 1, 224, 224)
    output = colornet(data)
    print(output)
    print(output[0].shape, output[1].shape)
