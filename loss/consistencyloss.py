import torch
import torch.nn as nn
from config import config
import transformer as tf
class ConsistencyLoss(nn.Module):
    def __init__(self, sampleNum = config.nSamplesconfi, cellGTSize = config.cellGTSize ,celldan = config.celldan):
        super(ConsistencyLoss, self).__init__()
        self.sampleNum = sampleNum
        self.cellGTSize = cellGTSize
        self.celldan= celldan
        self.block_partComposition = tf.partComposition()

    def forward(self,shape_rlt, trans_rlt, quat_rlt,batchSamplepoint,inUse):
        # get sample point
        samplePoint = batchSamplepoint.detach()
        # get tsdf
        #print(inUse)
        samplePoint = samplePoint.unsqueeze(1).expand(-1, config.primNum, -1, -1).cuda()
        tsdfOut , tsdfGT= self.block_partComposition(samplePoint,shape_rlt, trans_rlt, quat_rlt,inUse)
        #get ConsistencyLoss
        return tsdfOut, tsdfGT
