import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

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
    def __init__(self):
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
          nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
          nn.Upsample(scale_factor=2)
    )
    def forward(self, x):
        return self.upsample(x)


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.down_net = Backbone()
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
    print(output.shape)
