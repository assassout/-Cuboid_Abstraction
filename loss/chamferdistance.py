import torch
import torch.nn as nn
from config import config
import transformer as tf
from loss import samplePointMethod as sp
class ChamferDistance(nn.Module):
    def __init__(self, cellGTSize = config.cellGTSize ,celldan = config.celldan,np=config.primNum):
        super(ChamferDistance, self).__init__()
        self.cellGTSize = cellGTSize
        self.celldan= celldan
        self.np =np
        self.block_partComposition = tf.partComposition(primNum=np)
        self.block_sample = sp.SamplePoint(bs=config.batchSize,np=np,sampleNum=config.nSamplesChamfer,ifSurface = True)
        self.bolck_d1=nn.L1Loss(reduction='mean')
        self.bolck_d2=nn.L1Loss(reduction='mean')
    def forward(self,shape_rlt, trans_rlt, quat_rlt, CP,batchSamplepoint,inUse):
        # get sample point
        samplePoint = batchSamplepoint.detach()
        # get tsdf
        samplePoint = samplePoint.unsqueeze(1).expand(-1, self.np, -1, -1).cuda()
        tsdfOut , tsdfGT= self.block_partComposition(samplePoint,shape_rlt, trans_rlt, quat_rlt,inUse)
        #get ConsistencyLoss
        
        
        point = self.block_sample(shape_rlt, trans_rlt, quat_rlt)
        pointList = point.view(point.size(0),point.size(1)*point.size(2),point.size(3)).squeeze().contiguous()
        pointindex = pointList.add(0.5).mul(32.).int().clamp(min=0,max=31).detach().tolist()
        # get closed point(CP) bs X (np X nSample) X 3\
        #print(pointList.size())
        list_cp = CP.detach().tolist()
        closestPointList = pointList.detach().tolist()
        for i in range(pointList.size(0)): #bs
            j = 0
            while(j<pointList.size(1)): #np
                if inUse[i][j//config.nSamplesChamfer] == [1]:   #skip useless part
                    for k in range(j,j+config.nSamplesChamfer):
                        closestPointList[i][k]=list_cp[i][pointindex[i][k][0]][pointindex[i][k][1]][pointindex[i][k][2]]
                j += config.nSamplesChamfer
        CPlist = torch.Tensor(closestPointList).squeeze().cuda()
        # calculate loss
        pointList = pointList.view(point.size(0), point.size(1) , point.size(2), point.size(3)).squeeze().contiguous()
        CPlist = CPlist.view(point.size(0), point.size(1) , point.size(2), point.size(3)).squeeze()
        #pointList,CPlist 相减平方相加开根号
        pointList=(pointList - CPlist).pow(2).sum(3).pow(1/2)
        pointListGt = torch.zeros_like(pointList)
        #tsdfOut, tsdfGT 开根号相加
        tsdfOut = tsdfOut.pow(1/2)
        discd1=self.bolck_d1(pointList,pointListGt)
        discd2=self.bolck_d2(tsdfOut, tsdfGT)
        return discd1+discd2