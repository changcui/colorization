import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LowLevelFeatNet(nn.Module):
    def __init__(self):
        super(LowLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
       
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x1)))
        x2 = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x2)))
        x3 = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x3)))
        return x1, x2, x3, x


class MidLevelFeatNet(nn.Module):
    def __init__(self):
        super(MidLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, 
            padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.fusion1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.up2= nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, 
            padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fusion2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, 
            padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.fusion3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(2)
    def forward(self, x1, x2, x3, x):
        x = F.relu(self.bn1(self.up1(x)))
        x = torch.cat((x, x3), 1)
        x = F.relu(self.bn2(self.fusion1(x)))

        x = F.relu(self.bn3(self.up2(x)))
        x = torch.cat((x, x2), 1)
        x = F.relu(self.bn4(self.fusion2(x)))

        x = F.relu(self.bn5(self.up3(x)))
        x = torch.cat((x, x1), 1)
        x = F.relu(self.bn6(self.fusion3(x)))
        
        x = torch.sigmoid(self.bn7(self.conv(x)))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.down_net = LowLevelFeatNet()
        self.mid_net = MidLevelFeatNet()
        self.up_net = ColorizationNet()

    def forward(self, x):
        x1, x2, x3, x = self.down_net(x)
        x = self.mid_net(x)
        x = self.up_net(x1, x2, x3, x)
        return x


if __name__ == "__main__":

    colornet = ColorNet()
    data = torch.ones(3, 1, 224, 224)
    output = colornet(data)
    print(output)
    print(output.shape)
