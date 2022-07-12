'''
    File name: model.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import torch

import torch.nn as nn
import torch.nn.functional as F

from resnet import *


class FaceClassifier(nn.Module):
    '''
    '''
    def __init__(self, num_classes=7000, normalize_embeddings=True):
        '''
        '''
        super(FaceClassifier, self).__init__()

        # Normalization flag
        self.normalize_embeddings = normalize_embeddings

        # ResNet backbone
        self.backbone = ResNet50(img_channels=3, num_features=512)

        # Fully connected linear layer with normalized weights
        self.linear = nn.utils.weight_norm(nn.Linear(512, num_classes, bias=False), name='weight', dim=1)


    def forward(self, x, return_features=False):
        '''
            Input x: torch.tensor with shape batch_size x num_channels x height x width
        '''
        with torch.no_grad():
            self.linear.weight_g.copy_ = torch.ones_like(self.linear.weight_g)

        backbone_features = self.backbone(x)
        if self.normalize_embeddings:
            backbone_features = F.normalize(backbone_features, p=2, dim=1)

        if return_features:
            return backbone_features

        logits = self.linear(backbone_features)

        if self.normalize_embeddings:
            logits = torch.clamp(logits, min=-1.0, max=1.0)

        return logits
