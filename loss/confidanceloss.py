import torch
import torch.nn as nn
import math
from config import config
from loss import samplePointMethod as sp
class ConfidanceLoss(nn.Module):
    def __init__(self, sampleNum = config.nSamplesconfi, cellGTSize = config.cellGTSize ,celldan = config.celldan):
        super(ConfidanceLoss, self).__init__()
        self.sampleNum = sampleNum
        self.cellGTSize = cellGTSize
        self.celldan= celldan
        self.block_sample = sp.SamplePoint(bs=config.batchSize, np=config.primNum, sampleNum=config.nSamplesChamfer,ifSurface=False)
    def forward(self, shape_rlt, trans_rlt, quat_rlt, confi_rlt, batchVolume):
        # input bs x np x 1
        #batchVolume bs x 32 x 32 x 32
        # get near voxel to calculate GT (bs x np x 1 )
        voxel_volume = (1/32)**3
        volume_prim = shape_rlt.narrow(2,0,1) * shape_rlt.narrow(2,1,1)*shape_rlt.narrow(2,2,1) * 8

        volume_prim = volume_prim.squeeze().tolist()
        Volume = batchVolume.squeeze().detach().tolist()
        point = self.block_sample(shape_rlt, trans_rlt, quat_rlt)
        pointindex = point.add(0.5).mul(32.).int().clamp(min=0, max=31).detach().tolist()
        #print(batchVolume.size())
        pointList =point.tolist()
        IOUList =[]
        inUse = []
        judgeSize = self.celldan // 2
        for i in range(len(pointList)):
            IOUList.append([])
            inUse.append([])
            for j in range(len(pointList[i])):
                IOUList[i].append([])
                inUse[i].append([])
        for j in range(len(pointList[0])): # 0~63
            d_num =32 // self.celldan
            x = j % d_num * self.celldan + self.celldan // 2
            y = j // d_num % d_num * self.celldan + self.celldan // 2
            z = j // d_num // d_num * self.celldan + self.celldan // 2
            dlevel = [max(0, x - judgeSize), max(0, y - judgeSize),max(0, z - judgeSize)]
            hlevel = [min(31,x + judgeSize) + 1, min(31, y + judgeSize) + 1,min(31, z + judgeSize) + 1]
            for i in range(len(pointList)):
                # get voxelnum
                voxelnum = 0
                # intersect = 0
                for a in range(dlevel[0], hlevel[0]):
                    for b in range(dlevel[1], hlevel[1]):
                        for c in range(dlevel[2], hlevel[2]):
                            if (Volume[i][a][b][c] == 1):
                                voxelnum += 1
                                break
                #get point index
                if voxelnum == 0:
                    IOUList[i][j].append(0.)
                    inUse[i][j].append(0) #id useless part
                else:
                    IOUList[i][j].append(1)
                    inUse[i][j].append(1)
        IOUGT = torch.Tensor(IOUList).squeeze().cuda()
        confi_rlt =confi_rlt.squeeze()
        torch.set_printoptions(profile="full")
        #print(inUse[0])
        return confi_rlt, IOUGT ,inUse
