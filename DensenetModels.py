import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class AlexNet(nn.Module):
    def __init__(self, classCount, isTrained):
        super(AlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=isTrained)
        #kernelCount = self.vgg16.classifier.in_features
        #self.alexnet.classifier = nn.Sequential(nn.Linear(4096, classCount), nn.Sigmoid())
        self.alexnet.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classCount),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.alexnet(x)
        return x
#------------------------------------------------------------

class GoogLeNet(nn.Module):
    def __init__(self, classCount, isTrained):
        super(GoogLeNet, self).__init__()
        self.googlenet = torchvision.models.inception_v3(pretrained=isTrained)
        kernelCount = self.googlenet.fc.in_features
        print (kernelCount)
        self.googlenet.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.googlenet(x)
        return x

#-------------------------------------------------------------
class VGG16(nn.Module):
    def __init__(self, classCount, isTrained):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=isTrained)
        #kernelCount = self.vgg16.classifier.in_features
        self.vgg16.classifier = nn.Sequential(
         nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x
    
#---------------------------------------------------------------
class ResNet50(nn.Module):
    def __init__(self, classCount, isTrained):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)
        kernelCount = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
            
    def forward(self, x):
        x = self.resnet50(x)
        return x
    
#----------------------------------------------------------------
class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x
