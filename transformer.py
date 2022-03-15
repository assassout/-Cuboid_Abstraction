import torch.nn as nn
import torch

import quatUtils as qu
from config import config
import numpy as np
import torch.nn.functional as F
class rotation(nn.Module):
    def __init__(self,np,if_backward = False):
        self.np = np
        super(rotation, self).__init__()
        self.block_quatRotateModule = qu.quatRotateModule()
        self.if_backward=if_backward
    def forward(self, point, quat):
        zero = point.narrow(3,0,1).mul(0)
        pointsQuat = torch.cat((zero, point), 3)
        if not self.if_backward:
            quat_rev = torch.Tensor([-1,1,1,1]).cuda().view(1,1,4)
            quat_rev = quat_rev.expand(config.batchSize,self.np,-1)
            quat = quat*quat_rev
        quat_Rep = quat.unsqueeze(2).expand(-1,-1, point.size(2), -1)
        pointOut = self.block_quatRotateModule(pointsQuat, quat_Rep)
        return pointOut

class translation(nn.Module):
    def __init__(self,np):
        self.np = np
        super(translation, self).__init__()
    def forward(self, point, trans):
        trans_Rep = trans.unsqueeze(2).expand(-1, -1, point.size(2), -1)
        pointOut = point + trans_Rep
        return pointOut

class zoom(nn.Module):
    def __init__(self, np):
        self.np = np
        super(zoom, self).__init__()
    def forward(self, point, shape):
        shape_Rep = shape.unsqueeze(2).expand(-1, -1, point.size(2), -1)
        pointOut = point * shape_Rep
        return pointOut


class Transform():
    def Transform(self,point,shape,trans,quat,np=config.primNum,bs = config.batchSize):
        point = torch.FloatTensor(point).unsqueeze(0).unsqueeze(0).expand(bs, np, -1,-1).cuda()
        block_zoom = zoom(np)
        block_trans = translation(np)
        block_quat = rotation(np)
        p1 = block_zoom(point, shape)
        p2 = block_quat(p1, quat)
        p3 = block_trans(p2, trans)
        return p3.tolist()

class Transform_nn(nn.Module):
    def __init__(self, np=config.primNum,bs = config.batchSize):
        super(Transform_nn, self).__init__()
        self.np = np
        self.bs = bs
        self.block_zoom = zoom(np)
        self.block_trans = translation(np)
        self.block_quat = rotation(np)
    def forward(self,point,shape,trans,quat,if_mesh = False):
        if if_mesh:
            point = torch.FloatTensor(point).unsqueeze(0).unsqueeze(0).expand(self.bs, self.np, -1, -1).cuda()
        else:
            point = torch.FloatTensor(point).cuda()
        p1 = self.block_zoom(point, shape)
        p2 = self.block_quat(p1, quat)
        p3 = self.block_trans(p2, trans)
        return p3

#union for data and pred
class primUnion(nn.Module):
    def __init__(self,sp=config.nSamplePoints,np = config.primNum):
        super(primUnion, self).__init__()
        self.sp = sp
        self.np = np
        self.block_unionQuat = qu.Mult_quat()
    def forward(self, shape,trans,quat,deltashape,deltatrans,deltaquat,looptime = 1):
        shape_rlt = shape + deltashape
        trans_rlt = trans + deltatrans
        quat_rlt = self.block_unionQuat(quat, deltaquat)

        #print(shape_rlt)
        return shape_rlt, trans_rlt, quat_rlt

#tsdf for loss
# input is BX(np)  xsPX3 points, BX (np) X 3 part params
# output is BX nP X sp  tsdf^2 values
class cuboid(nn.Module):
    def __init__(self,sp,np):
        super(cuboid, self).__init__()
        self.sp = sp
        self.np = np
    def forward(self, point, dim,if_mutex = False):
        pAbs = point.abs()
        dimsRep = dim.unsqueeze(2).expand(-1, -1 ,point.size(2), -1).abs()
        if if_mutex == False:
            tsdfSq = pAbs.sub(dimsRep).relu().pow(2).sum(dim=3)
        else:
            tsdfSq,_ = dimsRep.sub(pAbs).relu().pow(2).min(dim=3)
        return tsdfSq


# input is BX(np)  xsPX3 points, BX (np) X 3 part params
# output is BX nP X sp  tsdf^2 values
class primitiveSelector(nn.Module):
    def __init__(self,sp=config.nSamplePoints,np = config.primNum):
        super(primitiveSelector, self).__init__()
        self.sp = sp
        self.np = np
        self.block_cuboid = cuboid(sp,np)
    def forward(self, point, shape,if_mutex = False):
        if if_mutex == False:
            temp_tsdf = self.block_cuboid(point, shape) #may need to transpose bs np
        else:
            temp_tsdf = self.block_cuboid(point, shape,if_mutex = True) #may need to transpose bs np
        tsdf_out = temp_tsdf.contiguous()
        return tsdf_out

class rigidTsdf(nn.Module):
    def __init__(self,np=config.primNum):
        super(rigidTsdf, self).__init__()
        self.block_trans = translation(np)
        self.block_rotation = rotation(np,if_backward=True)

    def forward(self, point, trans, quat):
        minus_t = trans.mul(-1.)
        p1 = self.block_trans(point, minus_t)
        #print(p1[0][0][0])
        p2 = self.block_rotation(p1, quat)
        #print(p2[0][0][0])
        return p2

#input points trans shape quat out tsdf
class tsdfTransform(nn.Module):
    def __init__(self,primNum = config.primNum):
        super(tsdfTransform, self).__init__()
        
        self.block_rigidTsdf = rigidTsdf(np = primNum)
        self.block_Selector = primitiveSelector(np = primNum)
    def forward(self, point, shape, trans, quat,if_mutex = False):
        pl = self.block_rigidTsdf(point, trans, quat)
        # bs x np x sp x 3
        if if_mutex == False:
            tsdf = self.block_Selector(pl, shape)
        else:
            tsdf = self.block_Selector(pl, shape,if_mutex = True)
        return tsdf

# input points trans shape quat out tsdf (xBs)
class partComposition(nn.Module):
    def __init__(self,primNum = config.primNum):
        super(partComposition, self).__init__()
        self.primNum = primNum
        self.block_tsdfTransform= tsdfTransform(primNum=primNum)
    def forward(self, point,shape, trans, quat,inUse):

        temp_tsdf = self.block_tsdfTransform(point,shape, trans, quat)

        # if if_all:
        #     tsdfout = temp_tsdf
        #     inUse = torch.Tensor(inUse).cuda().add(-1.).mul(-1)
        #     inUse = inUse.expand(inUse.size(0), inUse.size(1), point.size(2))
        #     temp_gt = (tsdfout * inUse).detach()
        # else:
        inUse = torch.Tensor(inUse).cuda().add(-1.).mul(-1.)  # useless part+1 useful part = org
        inUse = inUse.expand(inUse.size(0), inUse.size(1), point.size(2))
        tsdfout, index_tsdf = (temp_tsdf+inUse).min(dim = 1)
        temp_gt = torch.zeros_like(tsdfout)
        #tsdfout.expand(temp_tsdf.size(0), temp_tsdf.size(1)*temp_tsdf.size(2), 1).contiguous()
        return tsdfout,temp_gt

class partComposition_mutex(nn.Module):
    def __init__(self,primNum = config.primNum):
        super(partComposition_mutex, self).__init__()
        self.primNum = primNum
        self.block_tsdfTransform= tsdfTransform()
    def forward(self, point,shape, trans, quat,inUse,set1):
        
        temp_tsdf = self.block_tsdfTransform(point,shape, trans, quat,if_mutex = True)
        # if if_all:
        #     tsdfout = temp_tsdf
        #     inUse = torch.Tensor(inUse).cuda().add(-1.).mul(-1)
        #     inUse = inUse.expand(inUse.size(0), inUse.size(1), point.size(2))
        #     temp_gt = (tsdfout * inUse).detach()
        # else:
        inUse = torch.Tensor(inUse).cuda()
        inUse = inUse.expand(inUse.size(0), inUse.size(1), point.size(2))
        tsdfout= (temp_tsdf*inUse)
        #对应原基元加一
        tsdfout=tsdfout.mul(set1)
        #print(tsdfout)
        temp_gt = torch.zeros_like(tsdfout)
        #tsdfout.expand(temp_tsdf.size(0), temp_tsdf.size(1)*temp_tsdf.size(2), 1).contiguous()
        return tsdfout,temp_gt

# for beautify
class primitiveSelector_index(nn.Module):
    def __init__(self,sp=config.nSamplePoints,np = config.primNum):
        super(primitiveSelector_index, self).__init__()
        self.sp = sp
        self.np = np
        self.block_cuboid = cuboid(sp,np)
    def forward(self, point, shape):
        pAbs = point.abs()
        dimsRep = shape.unsqueeze(2).expand(-1, -1, point.size(2), -1).abs()
        DestAbs = pAbs.sub(dimsRep).abs()
        # bs x np x sp x 1(xyz index)
        _,xyz_index= DestAbs.min(dim = 3)
        return xyz_index

class partIndex(nn.Module):
    def __init__(self,primNum = config.primNum,batchSize=config.batchSize):
        super(partIndex, self).__init__()
        self.primNum = primNum
        self.batchsize=batchSize
        self.block_rigidTsdf = rigidTsdf()
        self.block_tsdfTransform= tsdfTransform()
        self.block_Selector = primitiveSelector_index()
    def forward(self, point,shape, trans, quat,inUse):
        temp_tsdf = self.block_tsdfTransform(point,shape, trans, quat)
        inUse = torch.Tensor(inUse).cuda()
        inUse= inUse.add(-1.).mul(-1.)  # useless part+1 useful part = org
        inUse = inUse.expand(inUse.size(0), inUse.size(1), point.size(2))
        _, index_tsdf = (temp_tsdf+inUse).min(dim = 1) # index_tsdf id belong to prim
        index_tsdf=index_tsdf.tolist()
        point_rigid = self.block_rigidTsdf(point, trans, quat)
        # bs x np x sp x 1(xyz index
        index_xyz = self.block_Selector(point_rigid,shape).tolist()
        shape_abs = shape.abs().tolist()
        point_rigid_abs = point_rigid.abs().tolist()
        point_rigid =point_rigid.tolist()
        # bs x np x 6 x n_in_face
        list_arrangedpoint=[]
        for bs in range(shape.size(0)):
            list_arrangedpoint.append([])
            for pr in range(self.primNum):
                list_arrangedpoint[bs].append([])
                for f in range(6):
                    list_arrangedpoint[bs][pr].append([])
        for bs in range(shape.size(0)):
                for v in range(config.nSamplePoints):
                    index_prim = index_tsdf[bs][v]
                    index_face = index_xyz[bs][index_prim][v] # now is only xyz index
                    if point_rigid[bs][index_prim][v][index_face] > 0:
                        index_face += 3
                    #print(index_face,point_rigid[bs][index_prim][v])
                    value_p_rigid = point_rigid_abs[bs][index_prim][v]
                    value_prim = shape_abs[bs][index_prim]
                    if value_p_rigid[0]< 1.5*value_prim[0] and value_p_rigid[0]> 0.5*value_prim[0] and \
                    value_p_rigid[1]< 1.5*value_prim[1] and value_p_rigid[1]> 0.5*value_prim[1] and \
                    value_p_rigid[2]< 1.5*value_prim[2] and value_p_rigid[2]> 0.5*value_prim[2] :
                        list_arrangedpoint[bs][index_prim][index_face].append(point_rigid[bs][index_prim][v])

        return list_arrangedpoint
