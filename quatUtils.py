import torch.nn as nn
import torch
class Mult_quat(nn.Module):
    def __init__(self):
        super(Mult_quat, self).__init__()
    def forward(self, q1, q2):
        q1_q2_prods = []
        inds = torch.IntTensor([0, 1, 2, 3,
                                1, 0, 3, 2,
                                2, 3, 0, 1,
                                3, 2, 1, 0]).reshape(4, 4).cuda()
        sign = torch.Tensor([1, -1, -1, -1,
                             1, 1, 1, -1,
                             1, -1, 1, 1,
                             1, 1, -1, 1]).reshape(4, 4).cuda()

        for d in range(4):
            q2_v1 = q2.narrow(2,inds[d][0], 1).mul(sign[d][0])
            q2_v2 = q2.narrow(2,inds[d][1], 1).mul(sign[d][1])
            q2_v3 = q2.narrow(2,inds[d][2], 1).mul(sign[d][2])
            q2_v4 = q2.narrow(2,inds[d][3], 1).mul(sign[d][3])
            q2sel = torch.cat((q2_v1, q2_v2, q2_v3, q2_v4),2)
            temp = torch.sum((q1 * q2sel), dim=2).unsqueeze(dim=2)
            q1_q2_prods.append(temp)
        qMult = torch.cat(q1_q2_prods,dim = 2)
        return qMult

class HamiltonProductModule(nn.Module):
    def __init__(self):
        super(HamiltonProductModule, self).__init__()
    def forward(self, q1, q2):
        q1_q2_prods = []
        inds = torch.IntTensor([0, 1, 2, 3,
                                1, 0, 3, 2,
                                2, 3, 0, 1,
                                3, 2, 1, 0]).reshape(4, 4).cuda()
        sign = torch.Tensor([1, -1, -1, -1,
                             1, 1, 1, -1,
                             1, -1, 1, 1,
                             1, 1, -1, 1]).reshape(4, 4).cuda()

        for d in range(4):
            q2_v1 = q2.narrow(3,inds[d][0], 1).mul(sign[d][0])
            q2_v2 = q2.narrow(3,inds[d][1], 1).mul(sign[d][1])
            q2_v3 = q2.narrow(3,inds[d][2], 1).mul(sign[d][2])
            q2_v4 = q2.narrow(3,inds[d][3], 1).mul(sign[d][3])
            q2sel = torch.cat((q2_v1, q2_v2, q2_v3, q2_v4),3)
            temp = torch.sum((q1 * q2sel), dim=3).unsqueeze(dim=3)
            q1_q2_prods.append(temp)
        qMult = torch.cat(q1_q2_prods,dim = 3)
        return qMult

class quatConjugateModule(nn.Module):
    def __init__(self):
        super(quatConjugateModule, self).__init__()
    def forward(self,x):
        x_split1 = x.narrow(3, 0, 1)
        x_split2 = x.narrow(3, 1, 3).mul(-1)
        xOut = torch.cat((x_split1,x_split2),3)
        return xOut

class quatRotateModule(nn.Module):
    def __init__(self):
        super(quatRotateModule, self).__init__()
        self.block_quatConjugate = quatConjugateModule()
        self.HamiltonProductModule = HamiltonProductModule()
    def forward(self, vec, quat):
        quat = quat.contiguous()
        quatConj = self.block_quatConjugate(quat)
        mult = self.HamiltonProductModule(self.HamiltonProductModule(quat,vec),quatConj)
        truncate =mult.narrow(3,1,3)
        return truncate

class quatToAngle(nn.Module):
    def __init__(self):
        super(quatToAngle, self).__init__()
    def forward(self,quat):
        q1 = quat.narrow(3, 0, 1).clone()
        q2 = quat.narrow(3, 1, 1).clone()
        q3 = quat.narrow(3, 2, 1).clone()
        q4 = quat.narrow(3, 3, 1).clone()
        x_angle = torch.atan2(2*(q1*q2+q3*q4),1-2*(torch.pow(q2,2)+torch.pow(q3,2)))
        y_angle = torch.asin(2*(q1*q3 -q2*q4))
        z_angle = torch.atan2(2*(q1*q4+q2*q3),1-2*(torch.pow(q3,2)+torch.pow(q4,2)))
        temp = torch.cat([x_angle,y_angle,z_angle], dim=3)
        return temp