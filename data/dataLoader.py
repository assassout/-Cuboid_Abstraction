import scipy.io as sio
import torch
from config import config
import numpy as np
import os

def loadFileData(fileName):
    #print(fileName)
    return sio.loadmat(fileName)

def getIndexTable(addr = config.dataDir, ext='mat'):
    fileList=[]
    if os.path.isdir(addr):
        for s in os.listdir(addr):
            subAddr = os.path.join(addr,s)
            fileList.append(subAddr)
    return fileList
def loadbatchDataOrg(indexTable , index,batchsize=config.batchSize):
    listTsdfGt = []
    listSamplePoint = []
    listCP = []
    listVolume = []

    for i in range(batchsize):
        temp = loadFileData(indexTable[(index + i) % len(indexTable)])
        listTsdfGt.append(torch.from_numpy(temp['tsdf']).unsqueeze(0))
        sp = temp['surfaceSamples']
        sp_less = []
        for i in range(10000):
            sp_less.append(sp[i])
        listSamplePoint.append(torch.Tensor(sp_less).unsqueeze(0))
        listCP.append(torch.from_numpy(temp['closestPoints']).unsqueeze(0))
        listVolume.append(torch.from_numpy(temp['Volume']).unsqueeze(0))
    batchVolume = torch.cat(listVolume, dim=0).unsqueeze(1)
    batchSamplepoint = torch.cat((listSamplePoint), dim=0)
    batchTsdf = torch.cat((listTsdfGt), dim=0)
    batchCP = torch.cat((listCP), dim=0)
    return batchVolume, batchTsdf, batchSamplepoint, batchCP
def loadbatchData(indexTable , index, loopTime ,batchsize=config.batchSize,matDir = config.matDir): #load batch data
    #load data base on index looptime = index // epoch
  
    batchVolume, batchTsdf, batchSamplepoint, batchCP =loadbatchDataOrg(indexTable , index)
    listVolumePrim = []
    listShapePrim = []
    listTransPrim = []
    listQuatPrim = []
    for i in range(batchsize):
        loopIndex = loopTime
        #load nameLoop 'primLooptime'
        if loopIndex == 0:
            dir1 = config.prim_intDir + "init"+str(config.primNum)+".mat"
        else:
            #loopIndex !=0 load data saved last loop
            (name_mat, _) = os.path.splitext(os.path.basename(indexTable[(index+i) % len(indexTable)]))
            name_mat = name_mat + 'loop' + str(loopIndex-1) + ".mat"
            dir1 = matDir + config.netName+'loop' +str(loopIndex-1)+'/'+ name_mat
        #print(dir1)

        temp = loadFileData(dir1)
        listVolumePrim.append(torch.from_numpy(temp['volume']).unsqueeze(0))
        listShapePrim.append(torch.from_numpy(temp['shape_rlt']).unsqueeze(0))
        listTransPrim.append(torch.from_numpy(temp['trans_rlt']).unsqueeze(0))
        listQuatPrim.append(torch.from_numpy(temp['quat_rlt']).unsqueeze(0))
    batchVolumePrim = torch.cat(listVolumePrim, dim=0).unsqueeze(dim=1)
    batchShape = torch.cat((listShapePrim), dim=0)
    batchTrans = torch.cat((listTransPrim), dim=0)
    batchQuat = torch.cat((listQuatPrim), dim=0)
    #print(batchQuat.size())
    return batchVolume, batchTsdf, batchSamplepoint,batchCP,batchVolumePrim,batchShape,batchTrans,batchQuat
    
def loadOneData(dataDir):
    listVolume =[]
    temp = loadFileData(dataDir)
    listVolume.append(torch.from_numpy(temp['Volume']).unsqueeze(0))
    batchCP = torch.cat((listVolume), dim=0)
    return batchCP
def loadVFData(indexTable):
    listVertics = []
    listFace=[]
    for i in range(len(indexTable)):
        temp = loadFileData(indexTable[i])
        listVertics.append(temp['vertices'])
        listFace.append(temp['faces'])
    return listVertics, listFace





