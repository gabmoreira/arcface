'''
    File name: resnet.py
    Author: Gabriel Moreira
    Date last modified: 07/07/2022
    Python Version: 3.9.13
'''

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"

        super(Block, self).__init__()
        self.num_layers = num_layers

        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1

        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample


    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_features):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()

        self.num_features = num_features

        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]

        self.in_channels = 64
        self.conv1       = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1         = nn.BatchNorm2d(64)
        self.relu        = nn.ReLU()
        self.maxpool     = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels= 64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # FC layer if we do not want the features
        self.fc  = nn.Linear(2048, self.num_features) # 512 if resnet18,34
        self.bnf = nn.BatchNorm1d(512, eps=1e-05)

        self.drop = nn.Dropout(p=0.1) #

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # Typical ResNet features are these ones
        x = torch.flatten(x, 1)

        # Additional linear layer added
        x = self.fc(x)
        x = self.bnf(x)
        x = self.drop(x) #

        return x


    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))

        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))

        self.in_channels = intermediate_channels * self.expansion # 256

        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again

        return nn.Sequential(*layers)



def ResNet18(img_channels=3, num_features=512):
    return ResNet(18, Block, img_channels, num_features)


def ResNet34(img_channels=3, num_features=512):
    return ResNet(34, Block, img_channels, num_features)


def ResNet50(img_channels=3, num_features=512):
    return ResNet(50, Block, img_channels, num_features)