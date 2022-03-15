import torch
import torch.nn as nn
from config import config
from loss import samplePointMethod as sp
import transformer as tf
class SymmetryLoss(nn.Module):
    def __init__(self,sampleNum=config.nSamplesChamfer):
        super(SymmetryLoss, self).__init__()
        self.sampleNum = sampleNum
        self.block_sample = sp.SamplePoint(bs=config.batchSize,np=config.primNum,sampleNum=config.nSamplesChamfer,ifSurface = True)
        self.block_sample_weight =sp.SamplePointWeight(bs=config.batchSize,np=config.primNum,sampleNum=config.nSamplesChamfer)
        self.block_partComposition = tf.partComposition()
    def forward(self, shape_rlt, trans_rlt, quat_rlt,IOUlist,inUse_sym):
        point = self.block_sample(shape_rlt, trans_rlt, quat_rlt)
        weight =self.block_sample_weight(shape_rlt,IOUlist)
        weight = weight.view(weight.size(0),weight.size(1)*weight.size(2),weight.size(3))
        pointRef = point.clone()
        pointRef.narrow(3,2,1).copy_(pointRef.narrow(3,2,1).mul(-1))
        pointRef =pointRef.view(pointRef.size(0),pointRef.size(1)*pointRef.size(2),pointRef.size(3))
        pointRef = pointRef.unsqueeze(1).expand(-1, config.primNum, -1, -1).cuda()
        tsdfOut, tsdfGT = self.block_partComposition(pointRef, shape_rlt, trans_rlt, quat_rlt, IOUlist)
        inUse_sym = inUse_sym.expand(-1,-1,config.nSamplesChamfer).reshape(-1,config.nSamplesChamfer*config.primNum).cuda()
        tsdfOut =tsdfOut.cuda()*inUse_sym
        return tsdfOut,weight.squeeze(),tsdfGT