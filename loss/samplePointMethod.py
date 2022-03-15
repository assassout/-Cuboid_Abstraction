import torch
import torch.nn as nn
from config import config
from transformer import Transform_nn as tf_n
class SampleUnitPoint(nn.Module):
    def __init__(self,bs=config.batchSize,np=config.primNum,sampleNum=config.nSamplesChamfer,ifSurface = True):
        super(SampleUnitPoint, self).__init__()
        self.sampleNum = sampleNum
        self.ifSurface = ifSurface
        self.bs = bs
        self.np = np
        self.nsp = sampleNum // 3
    def forward(self):
        coeffBernoulli = torch.empty(size=[self.bs, self.np, self.nsp, 3]).uniform_(0, 1).bernoulli()
        coeffBernoulli = 2 * coeffBernoulli - 1.
        coeff_w = torch.ones(size=[self.bs, self.np, self.nsp, 3]).uniform_(-1, 1)
        coeff_h = torch.ones(size=[self.bs, self.np, self.nsp, 3]).uniform_(-1, 1)
        coeff_d = torch.ones(size=[self.bs, self.np, self.nsp, 3]).uniform_(-1, 1)
        if(self.ifSurface):
            coeff_w.narrow(3, 0, 1).copy_(coeffBernoulli.narrow(3, 0, 1).clone())
            coeff_h.narrow(3, 1, 1).copy_(coeffBernoulli.narrow(3, 1, 1).clone())
            coeff_d.narrow(3, 2, 1).copy_(coeffBernoulli.narrow(3, 2, 1).clone())
        coeffs = torch.cat((coeff_w, coeff_h), dim=2).contiguous()
        coeffs = torch.cat((coeffs, coeff_d), dim=2).contiguous()

        return coeffs



#output bs X np X nSample X 3
class SamplePoint(nn.Module):
    def __init__(self,bs=config.batchSize,sampleNum=config.nSamplesChamfer,np=config.primNum,ifSurface = True):
        super(SamplePoint, self).__init__()
        self.sampleNum = sampleNum
        self.ifSurface =ifSurface
        self.bs = bs
        self.np = np
        self.block_SampleUnitPoint = SampleUnitPoint(self.bs,np,self.sampleNum,self.ifSurface)
        self.block_transform = tf_n(np)
    def forward(self,shape_rlt, trans_rlt, quat_rlt):
        unitPoint = self.block_SampleUnitPoint()
        unitPoint = self.block_transform(unitPoint,shape_rlt, trans_rlt, quat_rlt)
        return unitPoint


class SamplePointWeight(nn.Module):
    def __init__(self,bs=config.batchSize,sampleNum=config.nSamplesChamfer,np=config.primNum):
        super(SamplePointWeight, self).__init__()
        self.sampleNum = sampleNum
        self.bs = bs
        self.np = np
    def forward(self, shape_rlt,inUse):
        inUse = torch.Tensor(inUse)
        inUse =inUse.expand(-1,-1,3).cuda()
        shape_rlt = shape_rlt.abs().mul(inUse).add(1e-6)
        w = shape_rlt.narrow(2, 0, 1)
        h = shape_rlt.narrow(2, 1, 1)
        d = shape_rlt.narrow(2, 2, 1)
        area = 2*((w * h)+(h * d)+(w *d))
        areaRep = area.view(area.size(0),area.size(1),1,area.size(2))
        areaRep = areaRep.expand(-1,-1,self.sampleNum,-1)
        shapeInv = shape_rlt.pow(-1)
        shapeInvNorm = shapeInv.sum(2).unsqueeze(2).expand(-1,-1,3)
        normWeights = shapeInv.div(shapeInvNorm+1e-6).mul(3.)

        widthWt = normWeights.narrow(2, 0, 1).expand(-1,-1,self.sampleNum //3)
        heightWt = normWeights.narrow(2, 1, 1).expand(-1,-1,self.sampleNum//3)
        depthWt= normWeights.narrow(2, 2, 1).expand(-1,-1,self.sampleNum//3)
        finalWt = torch.cat([widthWt,heightWt,depthWt],2).unsqueeze(3)
        impWeights = (areaRep * finalWt).mul(1 / self.sampleNum)

        totWeights = impWeights.sum(2).add(1e-6)
        totWeights = totWeights.unsqueeze(2).expand(-1,-1,self.sampleNum,-1)
        normWeights_all = impWeights.div(totWeights)
        #print(normWeights_all)
        return normWeights_all