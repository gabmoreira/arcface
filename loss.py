'''
    File name: loss.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import torch
import numpy as np


class AngularLoss(torch.nn.Module):
    def __init__(self, radius=64.0, margin=5e-1):
        '''
            Since this inherits from nn.Module we do not need to define
            the backward pass. Autograd takes care of that for us.

            Input radius: float - radius of the hypersphere
            Input margin: float - force same class embeddings to be closer to class center
        '''
        super(AngularLoss, self).__init__()

        self.radius     = radius
        self.cos_margin = np.cos(margin)
        self.sin_margin = np.sin(margin)

        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def forward(self, logits, labels):
        '''
            Input logits: torch.tensor batch_size x num_classes
            Input labels: torch.tensor batch_size (numeric labels)
        '''
        target_mask = torch.nn.functional.one_hot(labels, 7000)

        # Since we have W_i . x = cos(theta) because ||W_i|| = 1 and ||x|| = 1
        # We can compute cos(theta+m) via cos(theta+m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        # We start by computing sin(theta) using sin(theta)^2 + cos(theta)^2 = 1
        sin_theta = torch.sqrt(1.0 - torch.pow(logits, 2))

        arc_logits_target = logits * self.cos_margin - sin_theta * self.sin_margin
        #arc_logits_target = logits - self.sin_margin

        arc_logits = self.radius * ( (1-target_mask)*logits + target_mask*arc_logits_target )

        loss = self.cross_entropy(arc_logits, labels)

        return loss
