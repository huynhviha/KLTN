import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer
  
#--------------------------------------------------------------------------------   

def runTrain(nnArchitecture, pathModel):
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = './dataset/ChestX-ray14/images'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/ChestX-ray14/annotations/train_1.txt'
    pathFileVal = './dataset/ChestX-ray14/annotations/val_1.txt'
    pathFileTest = './dataset/ChestX-ray14/annotations/test_1.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnIsTrained = False
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = pathModel + '.pth.tar'
    print ("pathModel: ", pathModel)
    
    start_time = time.time()
    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None, pathModel)
    end_time = time.time()
    print ("Time spent on training: ", end_time - start_time)

#-------------------------------------------------------------------------------- 

def runTest(nnArchitecture, pathModel):
    
    pathDirData = './dataset/ChestX-ray14/images'
    pathFileTest = './dataset/ChestX-ray14/annotations/test_1.txt'
    nnIsTrained = False
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = pathModel + '.pth.tar'
    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    Model = sys.argv[2]
    pathModel = sys.argv[3]
    if sys.argv[1] == '0':
        print("-----TRAIN-----")
        runTrain(Model, pathModel)
    else:
        print("-----TEST-----")
        runTest(Model, pathModel)

