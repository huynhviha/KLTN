#!/usr/bin/python
import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score, roc_curve, auc
from scipy import interp
from DensenetModels import VGG16
from DensenetModels import ResNet50
from DensenetModels import AlexNet
from DensenetModels import GoogLeNet
from DensenetModels import DenseNet121

from DatasetGenerator import DatasetGenerator

#-------------------------------------------------------------------------------- 

class ChexnetTrainer ():
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint, pathModel):

        
        # mô hình
        if nnArchitecture == 'DenseNet121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'VGG16': model = VGG16(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'GoogLeNet': model = GoogLeNet(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'AlexNet': model = AlexNet(nnClassCount, nnIsTrained).cuda()
            
        model = torch.nn.DataParallel(model).cuda()
                
        # xử lý dữ liệu
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        # load dataset
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
        
        # optimizer
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        # loss
        loss = torch.nn.BCELoss(size_average = True)
        
        # check point 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        # training
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.item())
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, pathModel)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss = ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss = ' + str(lossVal))
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        for batchID, (input, target) in enumerate (dataLoader):
                        
            target = target.cuda(async = True)
            input = input.cuda()
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        with torch.no_grad():
            for i, (input, target) in enumerate (dataLoader):

                target = target.cuda(async=True)
                input = input.cuda()     
                varInput = input
                varTarget = target   
                varOutput = model(varInput)
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor

                lossVal += losstensor.item()
                lossValNorm += 1
            
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
    
    # AUC
    
    def computeAUROC (dataGT, dataPRED, classCount):
        fpr = []
        tpr = []
        outAUROC = []
        thresholds = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            _fpr, _tpr, threshold = roc_curve(datanpGT[:, i], datanpPRED[:, i])
            fpr.append(_fpr)
            tpr.append(_tpr)
            thresholds.append(threshold)
        return outAUROC, fpr, tpr, thresholds

    #--------------------------------------------------------------------------------  
    # test 
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True
        
        # mô hình
        
        if nnArchitecture == 'DenseNet121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'VGG16': model = VGG16(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'ResNet50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'GoogLeNet': model = GoogLeNet(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'AlexNet': model = AlexNet(nnClassCount, nnIsTrained).cuda()
            
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
        model.load_state_dict(state_dict)
        print ("Loaded Checkpoint")

        # xử lý dữ liệu
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # load dataset
        
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                #print (i, input, target)
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                #print("GT", outGT)
                bs, n_crops, c, h, w = input.size()
                varInput = input.view(-1, c, h, w).cuda()

                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                outPRED = torch.cat((outPRED, outMean.data), 0)
                
                #print("PRED", outPRED)
        aurocIndividual, fpr, tpr, thresholds = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
            
        np.save('result/visualization/roc_auc.npy', aurocIndividual)
        np.save('result/visualization/fpr.npy', fpr)
        np.save('result/visualization/tpr.npy', tpr)
        np.save('result/visualization/threshold.npy', thresholds)
#-------------------------------------------------------------------------------- 