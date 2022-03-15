import torch
import torch.nn as nn
from config import config
from loss import samplePointMethod as sp
class CoverageLoss(nn.Module):
    def __init__(self,sampleNum=config.nSamplesChamfer,npin = config.primNum):
        super(CoverageLoss, self).__init__()
        self.sampleNum = sampleNum
        self.block_sample = sp.SamplePoint(bs=config.batchSize,np=npin,sampleNum=config.nSamplesChamfer,ifSurface = True)
        self.block_sample_weight = sp.SamplePointWeight(bs=config.batchSize, np=npin,sampleNum=config.nSamplesChamfer)
    def forward(self,shape_rlt, trans_rlt, quat_rlt, CP,IOUlist):
        # sample point
        point = self.block_sample(shape_rlt, trans_rlt, quat_rlt)
        weight = self.block_sample_weight(shape_rlt,IOUlist)
        pointList = point.view(point.size(0),point.size(1)*point.size(2),point.size(3)).squeeze().contiguous()
        pointindex = pointList.add(0.5).mul(32.).int().clamp(min=0,max=31).detach().tolist()
        # get closed point(CP) bs X (np X nSample) X 3\
        #print(pointList.size())
        list_cp = CP.detach().tolist()
        closestPointList = pointList.detach().tolist()
        for i in range(pointList.size(0)): #bs
            j = 0
            while(j<pointList.size(1)): #np
                if IOUlist[i][j//config.nSamplesChamfer] == [1]:   #skip useless part
                    for k in range(j,j+config.nSamplesChamfer):
                        closestPointList[i][k]=list_cp[i][pointindex[i][k][0]][pointindex[i][k][1]][pointindex[i][k][2]]
                j += config.nSamplesChamfer




        CPlist = torch.Tensor(closestPointList).squeeze().cuda()
        # calculate loss
        pointList = pointList.view(point.size(0), point.size(1) , point.size(2), point.size(3)).squeeze().contiguous()
        CPlist = CPlist.view(point.size(0), point.size(1) , point.size(2), point.size(3)).squeeze()
        return pointList,weight,CPlist

