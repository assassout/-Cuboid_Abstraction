import torch
from torch.autograd import Variable
from voxelnet import VNet
import torch.backends.cudnn
from data import dataLoader as dt
from config import config
from transformer import primUnion
from transformer import Transform_nn as tf
import mesh_ops
import os
from loss import confidanceloss

def saveLoopData(loopTime):
    fileIndexTable = dt.getIndexTable(config.dataDir)
    fileIndexTableTest = dt.getIndexTable(config.dataDir_t)
    if loopTime ==0:
        begin = 0
    else:
        begin = 1
    for LP in range(begin,loopTime+1):
        if LP == 0:
            config.ifShapeOrg = True  #
            config.shapeLrDecay = 0.01
            config.transLrDecay = 1
            net_num = config.loopInit
            bias_index =0
            #print(os.path.join(config.netDir, config.netName + 'loop' + "0") + '/' + str(net_num) + ".pth")
            net = torch.load(os.path.join(config.netDir, config.netName + 'loop' + "0") + '/' + str(net_num) + ".pth").cuda()
        else:
            config.ifShapeOrg = False
            #LDG.saveLoopData(0)
            config.shapeLrDecay = 0.01
            config.transLrDecay = 0.01
            bias_index = len(fileIndexTable) * config.Nepochs * LP
            bias_index_t = len(fileIndexTableTest) * config.Nepochs * LP
            #net_num = len(fileIndexTable) * config.Nepochs * (loopTime) // 32 -1
            net_num = config.numTrainIter - 1
            net = torch.load(os.path.join(config.netDir, config.netName + 'loop' + str(loopTime)) + '/' + str(net_num) + ".pth").cuda()
        block_union = primUnion()
        block_confiloss = confidanceloss.ConfidanceLoss()
        #load data
        for i in range(len(fileIndexTable) // config.batchSize +1):
            print("batch:"+str(i))
            index_begin = ( i * config.batchSize) % len(fileIndexTable)
            batchVolume, batchTsdf, batchSamplepoint, batchClosestPoints,batchVolumePrim,shape,trans,quat = dt.loadbatchData(fileIndexTable,bias_index+(i*32) % (config.numTrainIter*32), LP)
            batchVolume = Variable(batchVolume.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
            shape = Variable(shape.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
            trans = Variable(trans.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
            quat = Variable(quat.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
            batchVolumePrim = Variable(batchVolumePrim.type(torch.cuda.FloatTensor), requires_grad=False).cuda()
            if LP ==0:
                shape_dlt, trans_dlt, quat_dlt, confi_rlt = net(batchVolume, batchVolumePrim, shape, trans, quat)
                shape, trans, quat = block_union(shape, trans, quat, shape_dlt, trans_dlt, quat_dlt)
  
            else:
                for small_loop in range(config.stepNum):
                    shape_dlt, trans_dlt, quat_dlt, confi_rlt = net(batchVolume,batchVolumePrim,shape, trans, quat)
                    shape, trans, quat = block_union(shape, trans, quat, shape_dlt, trans_dlt, quat_dlt)
            #shape_rlt, trans_rlt, quat_rlt = shape, trans, quats
            _, _, inUse = block_confiloss(shape, trans, quat, confi_rlt, batchVolume)
            block_transform = tf()
            pointlist = block_transform(config.pointset,shape, trans, quat,if_mesh=True).tolist()
            mesh_ops.saveStepData(shape, trans, quat,fileIndexTable,index_begin,LP,confi_rlt.tolist(),pointlist,inUse)


def GenerateInitData():
    fileIndexTable = dt.getIndexTable(config.dataDir)
    batchVolume, batchTsdf, batchSamplepoint, batchClosestPoints, batchVolumePrim, shape, trans, quat = dt.loadbatchData(fileIndexTable, 0, 0)
    shape = Variable(shape.type(torch.cuda.FloatTensor), requires_grad=False).cuda()
    trans = Variable(trans.type(torch.cuda.FloatTensor), requires_grad=False).cuda()
    quat = Variable(quat.type(torch.cuda.FloatTensor), requires_grad=False).cuda()
    pointlist = tf.Transform(config.pointset, shape, trans, quat)
    confi = torch.zeros(32,27,1).tolist()
    inUse = torch.ones(32,27,1).tolist()
    mesh_ops.saveStepData(shape, trans, quat, fileIndexTable, 0, 0, confi,
                          pointlist, inUse)