'''
    File name: train.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as ttf

from resnet import ResNet18
from utils import getNumTrainableParams
from trainer import Trainer
from torch.utils.data import Dataset, DataLoader


if __name__ == '__main__':

    TRAIN_DIR  = "./classification/classification/classification/train"
    #TRAIN_DIR  = "./classification/train_subset/train_subset/train"
    DEV_DIR    = "./classification/classification/classification/dev"
    
    # Hyperparams
    RESUME     = True
    NAME       = 'rn18_v1'
    EPOCHS     = 100
    BATCH_SIZE = 256
    LR         = 0.1

    config = {'name'       : NAME,
              'epochs'     : EPOCHS,
              'batch_size' : BATCH_SIZE,
              'lr'         : LR,
              'resume'     : RESUME}

    print('Experiment ' + config['name'])

    # If experiment folder doesn't exist create it
    if not os.path.isdir(config['name']):
        os.makedirs(config['name'])
        print("Created experiment folder : ", config['name'])
    else:
        print(config['name'], "folder already exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        torch.cuda.empty_cache()

    train_transforms = [ttf.ToTensor(),
                        ttf.RandomHorizontalFlip(p=0.5),
                        ttf.RandomAdjustSharpness(sharpness_factor=2),
                        ttf.ColorJitter(brightness=.3, hue=.1, saturation=0.2),
                        ttf.RandomRotation(degrees=(-30, 30)),
                        ttf.RandomPerspective(distortion_scale=0.4, p=0.75)]

    dev_transforms = [ttf.ToTensor()]

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=ttf.Compose(train_transforms))
    dev_dataset   = torchvision.datasets.ImageFolder(DEV_DIR, transform=ttf.Compose(dev_transforms))
    train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    dev_loader    = DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    # Choose the model
    model = ResNet18().to(device)   

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * config['epochs']))
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model,
                      config['epochs'],
                      optimizer,
                      scheduler,
                      criterion,
                      train_loader, 
                      dev_loader,
                      device,
                      config['name'],
                      config['resume'])

    # Verbose
    print('Running on', device)
    print('Train - {} batches of {} images'.format(len(train_loader), config['batch_size']))
    print('  Val - {} batches of {} images'.format(len(dev_loader), config['batch_size']))
    print('Number of trainable parameters: {}'.format(getNumTrainableParams(model)))
    print(model)

    trainer.fit() 

