import torch
import torch.nn as nn
from config import config
from loss import samplePointMethod as sp
import transformer as tf
class MutexLoss(nn.Module):
    def __init__(self,sampleNum=config.nSamplesChamfer):
        super(MutexLoss, self).__init__()
        self.sampleNum = sampleNum
        self.block_sample = sp.SamplePoint(bs=config.batchSize,np=config.primNum,sampleNum=config.nSamplesChamfer,ifSurface = False)
        self.block_sample_weight =sp.SamplePointWeight(bs=config.batchSize,np=config.primNum,sampleNum=config.nSamplesChamfer)
        self.block_partComposition = tf.partComposition_mutex()
    def forward(self, shape_rlt, trans_rlt, quat_rlt,IOUlist):
        point = self.block_sample(shape_rlt, trans_rlt, quat_rlt)
        weight =self.block_sample_weight(shape_rlt,IOUlist)
        weight = weight.view(weight.size(0),weight.size(1)*weight.size(2),weight.size(3))
        pointRef = point.clone()
        pointRef =pointRef.view(pointRef.size(0),pointRef.size(1)*pointRef.size(2),pointRef.size(3))
        pointRef = pointRef.unsqueeze(1).expand(-1, config.primNum, -1, -1).cuda()
        set1 = torch.ones((config.batchSize,config.primNum,config.nSamplesChamfer*config.primNum)).tolist()
        for b in range(config.batchSize):
            for n in range(config.primNum):
                for p in range(config.nSamplesChamfer*config.primNum):
                    if p>=n*config.nSamplesChamfer and p<(n+1)*config.nSamplesChamfer:
                        set1[b][n][p] = 0
        set1=torch.Tensor(set1).cuda()
        tsdfOut, tsdfGT = self.block_partComposition(pointRef, shape_rlt, trans_rlt, quat_rlt, IOUlist,set1)
        
        weight = weight.squeeze()
        weight = weight.unsqueeze(1).expand(-1, config.primNum, -1).contiguous()
        return tsdfOut,weight.squeeze(),tsdfGT