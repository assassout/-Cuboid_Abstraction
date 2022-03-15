from voxelnet import VNet
import torch.backends.cudnn
from data import dataLoader as dt
from config import config
from transformer import Transform_nn as tf
from loss import coverageloss
from loss import confidanceloss
from loss import consistencyloss
from loss import Symmetryloss
from loss import steploss as Steploss
from loss import chamferdistance as Chamferdistance
from torch.autograd import Variable
from transformer import primUnion
import os
import shutil 
from config import config
import mesh_ops
import NMS
import os
from beautify import beautify
import loopDataGenerate as LDG
config.if_init = True
loadNetName = config.netName+'loop'
num = config.classIndex
# 03001627 chair   02691156 airplane    02924116 bus   02828884 bench   03636649 lamp   04256520 sofa   04530566 ship
testdir ="../data/"+num+"_t/"
orgObjDir="../shapenet/"+num+"/"


config.loopInit =20000 
print(loadNetName)
secloop = 15000
config.netName ="test"
prim_intDir = "./cache/"
netDir = "./cache/net_cache/"
meshDir = "./cache/mesh_cache/"
matDir = "./cache/mat_cache/"
looptime = 1


FinalDir = "../result/"

loadNetName0 = loadNetName+'0'
fileIndexTable = dt.getIndexTable(testdir)
config.datasize = len(fileIndexTable)
testsize=config.datasize

#print(config.datasize)
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
block_union = primUnion()

block_covloss = coverageloss.CoverageLoss()
block_confiloss = confidanceloss.ConfidanceLoss()
block_consisloss = consistencyloss.ConsistencyLoss()
block_symloss = Symmetryloss.SymmetryLoss()
block_loss_confi = torch.nn.MSELoss(reduction='mean')
block_loss_cov = torch.nn.MSELoss(reduction='none')
block_loss_consis = torch.nn.L1Loss(reduction='mean')
block_loss_sym = torch.nn.L1Loss(reduction='none')
block_loss_step = Steploss.StepLoss(if_init=config.if_init)
Block_chamferdistance = Chamferdistance.ChamferDistance()
block_transform = tf()
#net.load_state_dict(Datanet)
#load data
def calculateLoss(shape, trans, quat,b,Lt,batchCP):
    confiOut, IOUGT, inUse = block_confiloss(shape, trans, quat, confi, batchVolume)
    pointOut_cov, weight_cov, CPlist = block_covloss(shape, trans, quat, batchClosestPoints, inUse)
    tsdfOut, tsdfGT = block_consisloss(shape, trans, quat, batchSamplepoint, inUse)
    confiloss = block_loss_confi(confiOut, IOUGT)
    consisloss = block_loss_consis(tsdfOut, tsdfGT)
    covloss = (block_loss_cov(pointOut_cov, CPlist) * weight_cov).sum(3).sum(2).sum(1).sum(
        0) / 3. / config.primNum / config.batchSize
    losstemp = config.consisWt * consisloss \
               + config.covWt * covloss
    ChamferDistance = Block_chamferdistance(shape, trans, quat, batchClosestPoints,batchSamplepoint,inUse)
    pointlist = block_transform(config.pointset, shape, trans, quat, if_mesh=True).tolist()
    IOU_fin,inUse_New = NMS.evaluation(shape, trans, quat, batchVolume, inUse,batchCP)
    IOU_eva = 0
    mesh_ops.saveStepData(shape, trans, quat, fileIndexTable, b*32, Lt, confi.tolist(), pointlist, inUse)
    if(Lt ==3):
        inUse_o = torch.Tensor(inUse_new).int()
        inUse_o=inUse_o.view(inUse_o.size(0), 8, 1).tolist()
        mesh_ops.saveStepData(shape, trans, quat, fileIndexTable, b*32, Lt+1, confi.tolist(), pointlist, inUse_o)
    for w in range(32):
        IOU_eva += IOU_fin[w] / 32.
    return ChamferDistance,losstemp,IOU_eva,inUse_New

loss = 0
IOU_a =0
dcd =0
loss_fin = 0
IOU_b = 0
dcd_fin =0
bs = config.datasize // config.batchSize +1
for b in range(bs):
    begin = 0
    shape_n,trans_n,quat_n = 0,0,0
    batchVolumePrim_n = 0
    for i in range(begin, 2):
        #print("bs:"+str(bs))
        if i == 0:
            config.ifShapeOrg = True
            config.shapeLrDecay = 0.001
            config.transLrDecay = 1
            net = torch.load(config.netDir + loadNetName0 + '/' + str(config.loopInit) + ".pth")
            #bus 19999 chair 31219 plane 26969
        else:
            config.ifShapeOrg = False
            config.shapeLrDecay = 0.01
            config.transLrDecay = 0.01
            net = torch.load(config.netDir + loadNetName+str(looptime)+ '/' + str(secloop) + ".pth")
        batchVolume, batchTsdf, batchSamplepoint, batchClosestPoints,batchVolumePrim,shape, trans, quat = dt.loadbatchData(fileIndexTable,b*32,i)
        if i !=0:
            shape, trans, quat = shape_n,trans_n,quat_n
        batchVolume = Variable(batchVolume.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
        batchVolumePrim = Variable(batchVolumePrim.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
        shape = Variable(shape.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
        trans = Variable(trans.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
        quat = Variable(quat.type(torch.cuda.FloatTensor),requires_grad=False).cuda()
        shape_dlt_all, trans_dlt_all, quat_dlt_all =[],[],[]
        if i == 0:
            shape_dlt, trans_dlt, quat_dlt, confi = net(batchVolume, batchVolumePrim, shape, trans, quat)
            shape, trans, quat = block_union(shape, trans, quat, shape_dlt, trans_dlt, quat_dlt)
            dis,losstemp,IOU_eva,inUse_new = calculateLoss(shape, trans, quat,b,0,batchClosestPoints)
            print("loopTime:" + str(i) + " BS:" + str(b) + ' ' + 'dcd: {:.6f}'.format(dis.tolist()) + ' ' + 'loss: {:.6f}'.format(losstemp.tolist()) + ' ' + 'IOU: {:.6f}'.format(IOU_eva))
            dcd += dis.tolist()
            loss += losstemp.tolist()
            IOU_a += IOU_eva
            inUse_new = torch.Tensor(inUse_new)
            inUse_new=inUse_new.view(inUse_new.size(0), 8, 1).tolist()
            shape_n,trans_n,quat_n = shape.clone(), trans.clone(), quat.clone()
        else:
            for s_loop in range(config.stepNum):
                shape_dlt, trans_dlt, quat_dlt, confi = net(batchVolume, batchVolumePrim, shape, trans, quat)
                #print(trans_dlt)
                shape, trans, quat = block_union(shape, trans, quat, shape_dlt, trans_dlt, quat_dlt)
                if s_loop == config.stepNum-1:
                    dis,losstemp, IOU_eva,inUse_new = calculateLoss(shape, trans, quat,b,1+s_loop,batchClosestPoints)
                    print("loopTime:" + str(i+s_loop) + " BS:" + str(b) + ' ' + 'dcd: {:.6f}'.format(dis.tolist()) + ' ' + 'loss: {:.6f}'.format(losstemp.tolist()) + ' ' + 'IOU: {:.6f}'.format(IOU_eva))
                    inUse_new = torch.Tensor(inUse_new)
                    inUse_new=inUse_new.view(inUse_new.size(0), 8, 1).tolist()
                    loss_fin += losstemp.tolist()
                    dcd_fin += dis.tolist()
                    IOU_b += IOU_eva


if not os.path.exists(FinalDir+loadNetName):
    os.mkdir(FinalDir+loadNetName) 
testDir = ['./cache/mat_cache/testloop0/','./cache/mat_cache/testloop1/','./cache/mat_cache/testloop3/','./cache/mat_cache/testloop4/',]
for root, dirs, files in os.walk(testdir):
    for f in files:
        print(f.split('.')[0])
        name = f.split('.')[0]
        if not os.path.exists(FinalDir+loadNetName+"/"+name):
            os.mkdir(FinalDir+loadNetName+"/"+name)
        file1 = orgObjDir +name+'/model.obj'
        file1_re = FinalDir+loadNetName+"/"+name+'/'+name+".obj"
        file2 = testDir[0] +name + 'loop0.obj'
        file2_re = FinalDir+loadNetName+"/"+name+'/'
        file3 = testDir[2] +name + 'loop3.obj'
        file3_re = FinalDir+loadNetName+"/"+name+'/'
        file4 = testDir[3] +name + 'loop4.obj'
        file4_re = FinalDir+loadNetName+"/"+name+'/'
        
        file2_mtl = testDir[0] +name + 'loop0.mtl'
        file2_remtl = FinalDir+loadNetName+"/"+name+'/'
        file3_mtl = testDir[2] +name + 'loop3.mtl'
        file3_remtl = FinalDir+loadNetName+"/"+name+'/'
        file4_mtl = testDir[3] +name + 'loop4.mtl'
        file4_remtl = FinalDir+loadNetName+"/"+name+'/'
        shutil.copy(file1,file1_re)
        shutil.copy(file2,file2_re)
        shutil.copy(file3,file3_re)
        shutil.copy(file4,file4_re)
        shutil.copy(file2_mtl,file2_remtl)
        shutil.copy(file3_mtl,file3_remtl)
        shutil.copy(file4_mtl,file4_remtl)
print("step1:")
print("loss:"+str(loss/(bs+1))+" DCD:"+str(dcd/(bs+1))+" IOU:"+str(IOU_a/(bs+1)))
print("step2:")
print("loss:"+str(loss_fin/(bs+1))+" DCD:"+str(dcd_fin/(bs+1))+" IOU:"+str(IOU_b/(bs+1)))  
#os.system('rm %s'%(testDir[0]))
#os.system('rm %s'%(testDir[2]))
#os.system('rm %s'%(testDir[3]))
























