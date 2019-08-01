import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import time
import sys
import pandas as pd
import scipy as sp
from IPython.display import display
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from DensenetModels import DenseNet121

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
          
        import re
        pattern = re.compile(
             r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
       
        model = torch.nn.DataParallel(model).cuda() 
        modelCheckpoint = torch.load(pathModel)
        state_dict = modelCheckpoint['state_dict']
        
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
                #print (new_key)
        model.load_state_dict(state_dict)

        self.model = model.module.densenet121
        self.model.eval()
        
        
        #---- Initialize the weights
        self.weights = list(self.model.features.parameters())[-2]
        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.model.features.cuda()
        output = self.model.features(input.cuda())
        pred = self.model(input.cuda())
        
        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.5 + imgOriginal
        
        cv2.imwrite(pathOutputFile, img)
        return pred
#-------------------------------------------------------------------------------- 
if __name__ == '__main__':
    pathInputImage = sys.argv[1]
    name, form = pathInputImage.split('.')
    pathOutputImage = name + '_heatmap.' + form
    pathModel = 'models/m-25012018-123527.pth.tar'
    img = cv2.imread(pathInputImage)
    
    nnArchitecture = 'DENSE-NET-121'
    nnClassCount = 14
    transCrop = 224
    
    FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    m, n = pathInputImage.split("/")
    n = n + ".png"
    labels = open("label.txt", "r+")
    for label in labels:
        l = label.split(" ")
        a, b = l[0].split("/")
        if b in n:
            GT = np.zeros(14)
            for i in range(14):
                GT[i] = int(l[i + 1])
                
    h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
    pred = h.generate(pathInputImage, pathOutputImage, transCrop)
    pred = pred.cpu()
    
    threshold = np.load("best_threshold.npy")
    pred_bi = np.zeros(14)
    for i in range(14):
        if pred[0][i] >= threshold[i]:
            pred_bi[i] = 1
    
    preds_concat=pd.concat([pd.Series(FINDINGS), pd.Series(pred_bi), pd.Series(GT)], axis=1)
    preds = pd.DataFrame(data=preds_concat)
    preds.columns=["Finding", "Predicted Binary", "Ground Truth"]
    preds.set_index("Finding",inplace=True)
    #preds.sort_values(by='Predicted Probability',inplace=True,ascending=False)
    display(preds)
    #preds.sort_values(by='Predicted Probability',inplace=True,ascending=False)
    accuracy = 0
    for i in range(14):
        if GT[i] == pred_bi[i]:
            accuracy = accuracy + 1
    print ("Threshold: ", threshold)
    print ("Prediction: ", pred)
    print ("# true prediction: ", accuracy)
    print ("Accuracy: ", accuracy/14)
        