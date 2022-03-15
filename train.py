import torch.nn as nn
import torch
from torch.autograd import Variable
import time
from voxelnet import VNet
import loopDataGenerate as LDG
import torch.nn.init as init
import torch.backends.cudnn
from loss import samplePointMethod as sp
from data import dataLoader as dt
from config import config
from transformer import Transform_nn as tf
from transformer import tsdfTransform
from transformer import primUnion
import torch.nn.functional as F 
import os
from loss import steploss as Steploss
from loss import coverageloss
from loss import confidanceloss
from loss import consistencyloss
from loss import Symmetryloss
from loss import mutexloss as Mutexloss
import mesh_ops
import sys
#set devices
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
torch.set_printoptions(precision=8)

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

def get_symlOU(shape,trans,quat,inUse):
    # bs x (np xsp) x 3
    block_sample = sp.SamplePoint(bs=config.batchSize,np=config.primNum,sampleNum=150,ifSurface = False)
    #get ConsistencyLoss
    point = block_sample(shape, trans, quat)
    point.narrow(3,2,1).copy_(point.narrow(3,2,1).mul(-1))
    point= point.view(point.size(0), point.size(1) * point.size(2), point.size(3))
    point = point.unsqueeze(1).expand(-1, config.primNum, -1, -1).cuda()
    bolck_tsdfTransform =tsdfTransform()
    #bs x np x (np xsp) x 3
    tsdf = bolck_tsdfTransform(point, shape, trans, quat)
    tsdf=tsdf.tolist()
    IOU_new=[]
    for bs in range(shape.size(0)):
        IOU_new.append([])
        for i in range(config.primNum):
            IOU_new[bs].append([])
            j=i
            if i==j:
                temp = 0
                for spn in range(150):
                    if tsdf[bs][j][150*i+spn] == 0:    #point sample on i tsdf on j    
                        temp += 1./150 
                if temp>0.3:
                    IOU_new[bs][j].append(0)
                elif temp<=0.3 and inUse[bs][j][0] ==0:
                    IOU_new[bs][j].append(0)
                else:
                    IOU_new[bs][j].append(1)
    return torch.Tensor(IOU_new)
    
# load Net
def loadNet():
    if config.ifLoadNet:
        net_loaddata=torch.load(config.netDir + config.loadNetName + '/' + str(config.loadNetIter) + ".pth").state_dict()
        net = VNet()
        net.apply(weights_init)
        print('Initializing weights complete')
        net_loadeddata= {}

        for k,v in net_loaddata.items():
            if k[0:7] == 'block_1' or k[0:7] == 'block_2' or k[0:7] == 'block_3' or k[0:7] == 'block_4' or k[0:7] == 'block_5':
                #print(k)
                net_loadeddata[k] = v
        net_data = net.state_dict()
        net_data.update(net_loadeddata)
        net.load_state_dict(net_data)
        #print(net_data)
        net.cuda()
        #pretrained_dict = {k: v for k, v in net_data.items() if k in net}
        #net.load_state_dict(net_data.state_dict(),strict=False)
    else:
        net = VNet()
        net.apply(weights_init)
        print('Initializing weights complete')
        net.cuda()
    return net
def get_optimizer(net,lptime):  
    # define optimizer set encoder lr to 0.001 decoder to 0.005
    if lptime ==0:
        LRa = config.LROrg
        LRb = config.LROrg
    else:
        LRa = config.learningRatePretrain
        LRb = config.learningRateDecoder
    conv1_params = list(map(id, net.block_1.parameters()))
    conv2_params = list(map(id, net.block_2.parameters()))
    conv3_params = list(map(id, net.block_3.parameters()))
    conv4_params = list(map(id, net.block_4.parameters()))
    conv5_params = list(map(id, net.block_5.parameters()))
    base_params = filter(lambda p: id(p) not in conv1_params+conv2_params +conv3_params+conv4_params + conv5_params,
                         net.parameters())
    optimizer = torch.optim.Adam([{'params': base_params},
                                  {'params': net.block_1.parameters(), 'lr': LRa},
                                  {'params': net.block_2.parameters(), 'lr': LRa},
                                  {'params': net.block_3.parameters(), 'lr': LRa},
                                  {'params': net.block_4.parameters(), 'lr': LRa},
                                  {'params': net.block_5.parameters(), 'lr': LRa},
                                  ],LRb, betas=(0.9, 0.999))
    # load loss module
    return optimizer

def train():
    #init
    # load data index
    fileIndexTable = dt.getIndexTable(config.dataDir)
    config.datasize = len(fileIndexTable)
    config.meshSaveIter = 1000
    config.netSaveIter = 500
    epoch_size = len(fileIndexTable)
    block_covloss = coverageloss.CoverageLoss()
    block_confiloss = confidanceloss.ConfidanceLoss()
    block_consisloss = consistencyloss.ConsistencyLoss()
    block_symloss = Symmetryloss.SymmetryLoss()
    block_steploss = Steploss.StepLoss(if_init=config.if_init)
    block_mutexloss = Mutexloss.MutexLoss()
    block_loss_confi = torch.nn.MSELoss(reduction='mean')
    block_loss_cov = torch.nn.MSELoss(reduction='none')
    block_loss_consis = torch.nn.L1Loss(reduction='mean')
    block_loss_mutex = torch.nn.L1Loss(reduction='none')
    block_loss_sym = torch.nn.L1Loss(reduction='none')
    block_union = primUnion()
    print(config.netName)
    # if config.if_init:
    #     begin = 0
    # else:
    #     begin = 1
    #LDG.saveLoopData(0)
    for Lt in range(0,2):
        # Lt = 0 , calculate init pos and size
        if Lt == 0:
            config.ifLoadNet = False
            config.ifShapeOrg = True   #
            config.learningRatePretrain = 0.001
            config.shapeLrDecay = 0.01
            config.transLrDecay = 1
            net = loadNet()
            net.train()
            optimizer = get_optimizer(net,Lt)
            #config.numTrainIter = config.datasize * config.Nepochs  // 32
            config.numTrainIter = config.loopInit+1
            bias_index = 0 #load int   data= 0  ~  config.datasize * config.Nepochs//32
        elif Lt == 1:
            config.ifLoadNet = True #load LT = 0's net
            config.loadNetName = config.netName +'loop0'
            config.loadNetIter = config.loopInit
            print(config.loopInit)
            config.ifShapeOrg = False
            config.learningRatePretrain = 0.0001
            #LDG.saveLoopData(0)
            config.shapeLrDecay = 0.01
            config.transLrDecay = 0.01
            net = loadNet()
            net.train()  #reload net
            optimizer = get_optimizer(net,Lt)
            config.numTrainIter = 20000
            #config.numTrainIter = config.datasize * config.Nepochs * (Lt)// 32
            bias_index = epoch_size * config.Nepochs  # not load init  data=bias_index  ~  config.datasize * config.Nepochs * (Lt) // 32 +bias_index

        if Lt !=0:
            netold = torch.load(config.netDir + config.netName+'loop0' + '/' + str(config.loopInit) + ".pth")
        print(config.numTrainIter)
        for iteration in range(config.numTrainIter):
            #load batch
            #iteration = 420
            t0 = time.time()
            config.ifShapeOrg = True
            config.shapeLrDecay = 0.001
            config.transLrDecay = 1
            _, _, _, _,batchVolumePrim_old,shape_old,trans_old,quat_old = dt.loadbatchData(fileIndexTable,(iteration*32)%len(fileIndexTable),0)
            batchVolume, batchTsdf, batchSamplepoint, batchClosestPoints, batchVolumePrim, _, _, _ = dt.loadbatchData(fileIndexTable, (iteration * 32), Lt)
            batchVolume = Variable(batchVolume.type(torch.cuda.FloatTensor)).cuda()
            batchVolumePrim_old = Variable(batchVolumePrim_old.type(torch.cuda.FloatTensor)).cuda()
            shape_old = Variable(shape_old.type(torch.cuda.FloatTensor)).cuda()
            trans_old = Variable(trans_old.type(torch.cuda.FloatTensor)).cuda()
            quat_old = Variable(quat_old.type(torch.cuda.FloatTensor)).cuda()
            if Lt !=0:
                shape_dlt_old, trans_dlt_old, quat_dlt_old, _= netold(batchVolume,batchVolumePrim_old,shape_old, trans_old, quat_old)
                shape, trans, quat = block_union(shape_old, trans_old, quat_old, shape_dlt_old, trans_dlt_old, quat_dlt_old)
            else:
                shape, trans, quat=shape_old.clone(), trans_old.clone(), quat_old.clone()
            
            
            #batchVolume, batchTsdf, batchSamplepoint, batchClosestPoints = dt.loadbatchDataOrg(fileIndexTable, (iteration*32)% config.numTrainIter)
            # forward
            #batchVolume =torch.cat((batchVolume,batchVolumePrim),dim=1)
            
            batchVolumePrim=Variable(batchVolumePrim.type(torch.cuda.FloatTensor)).cuda()
            shape = Variable(shape.type(torch.cuda.FloatTensor)).cuda()
            trans = Variable(trans.type(torch.cuda.FloatTensor)).cuda()
            quat = Variable(quat.type(torch.cuda.FloatTensor)).cuda()
            shapeorg = shape.clone()
            transorg = trans.clone()
            quatorg = quat.clone()
            shape_dlt_all = []
            quat_dlt_all = []
            trans_dlt_all = []
            #size [bs 1 32 32 32]
            if Lt !=0:
                config.ifShapeOrg = False
                config.shapeLrDecay = 0.01
                config.transLrDecay = 0.01
                for loop_in in range(config.stepNum):
                    shape_dlt, trans_dlt, quat_dlt, confi_rlt= net(batchVolume,batchVolumePrim,shape, trans, quat)
                    shape_dlt_all.append(shape_dlt.unsqueeze(dim=0))
                    trans_dlt_all.append(trans_dlt.unsqueeze(dim=0))
                    quat_dlt_all.append(quat_dlt.unsqueeze(dim=0))
                    shape, trans, quat = block_union(shape, trans, quat, shape_dlt, trans_dlt, quat_dlt)
                shape_dlt_all = torch.cat((shape_dlt_all),dim=0)
                trans_dlt_all = torch.cat((trans_dlt_all), dim=0)
                quat_dlt_all = torch.cat((quat_dlt_all), dim=0)
                #print(trans_dlt_all[0][0][0],quat_dlt_all[0][0][0])
            else:
                config.ifShapeOrg = True
                config.shapeLrDecay = 0.001
                config.transLrDecay = 1
                shape_dlt, trans_dlt, quat_dlt, confi_rlt = net(batchVolume,batchVolumePrim,shape, trans, quat)
                shape, trans, quat = block_union(shape, trans, quat, shape_dlt, trans_dlt, quat_dlt)
            confiOut, IOUGT ,inUse = block_confiloss(shape, trans, quat, confi_rlt, batchVolume)
            t3 = time.time()
            pointOut_cov,weight_cov,CPlist = block_covloss(shape, trans, quat, batchClosestPoints,inUse)
            t4 = time.time()
            tsdfOut, tsdfGT = block_consisloss(shape, trans, quat,batchSamplepoint,inUse)
            if(Lt==0):
                inUse_ten = torch.Tensor(inUse).cuda()
                tsdfOut_s, weight_sym,tsdfGT_s = block_symloss(shape, trans, quat, inUse,inUse_ten)
            else:
                inUse_sym=get_symlOU(shape,trans,quat,inUse)
                tsdfOut_s, weight_sym,tsdfGT_s = block_symloss(shape, trans, quat,inUse, inUse_sym)
            tsdfOut_m, weight_m,tsdfGT_m = block_mutexloss(shape, trans, quat, inUse)
            
            confiloss = block_loss_confi(confiOut, IOUGT)
            consisloss = block_loss_consis(tsdfOut, tsdfGT)
            covloss = (block_loss_cov(pointOut_cov,CPlist)*weight_cov).sum(3).sum(2).sum(1).sum(0) / 3. / config.primNum /config.batchSize
            symloss = (block_loss_sym(tsdfOut_s, tsdfGT_s)*weight_sym).sum(1).sum(0) / config.primNum /config.batchSize
            #mutexloss = (block_loss_mutex(tsdfOut_m, tsdfGT_m)*weight_m).sum(2).sum(1).sum(0)/ config.primNum / config.primNum /config.batchSize
            if Lt !=0:
                steploss = block_steploss(shape_dlt_all,trans_dlt_all,quat_dlt_all)
                loss = config.consisWt * consisloss \
                       +config.covWt *covloss\
                       +config.symWt * symloss\
                       +steploss #+mutexloss 
            else:
                loss = config.consisWt * consisloss \
                       +config.covWt *covloss\
                       +config.symWt * symloss\
                       #+mutexloss 
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(str(iteration+1)+' '+str(loss_confi)+' '+str(loss_coverage)+' '+str(loss_consistency))
            # ops for saving net
            if (iteration % config.netSaveIter == 0 or iteration == config.numTrainIter-1):
                if (not os.path.exists(os.path.join(config.netDir, config.netName+'loop'+str(Lt)))):
                    os.mkdir(os.path.join(config.netDir, config.netName+'loop'+str(Lt)))
                print("net saved at iteration" + str(iteration))
                torch.save(net, os.path.join(config.netDir, config.netName+'loop'+str(Lt))+'/'+ str(iteration)+".pth")
            t1 = time.time()
            #print information
            if Lt !=0:
                print("loopTime:"+str(Lt)+" iter:" + str(iteration) +' '+'Timer: %.4f sec.' % (t1 - t0)+' '+ 'loss: {:.6f}'.format(loss)+' '+ 'steploss: {:.6f}'.format(steploss)+' '+ 'consisloss: {:.9f}'.format(config.consisWt* consisloss)+' '+ 'covloss: {:.9f}'.format(covloss)+' '+ 'symloss: {:.9f}'.format(symloss))
            else:
                print("loopTime:"+str(Lt)+" iter:" + str(iteration) +' '+'Timer: %.4f sec.' % (t1 - t0)+' '+ 'loss: {:.6f}'.format(loss)+' '+ 'consisloss: {:.9f}'.format(config.consisWt* consisloss)+' '+ 'covloss: {:.9f}'.format(covloss)+' '+ 'symloss: {:.9f}'.format(symloss))
            if (iteration  == config.numTrainIter-1):
                print("loop:"+str(Lt))
                LDG.saveLoopData(Lt)

if __name__ == '__main__':

    train()
