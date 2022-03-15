from config import config
from loss import samplePointMethod as sp
import transformer as tf
import numpy
threshold = 0.6
def NMS(point,shape,trans,quat):
    # bs x (np xsp) x 3
    point= point.view(point.size(0), point.size(1) * point.size(2), point.size(3))
    point = point.unsqueeze(1).expand(-1, config.primNum, -1, -1).cuda()
    bolck_tsdfTransform =tf.tsdfTransform()
    #bs x np x (np xsp) x 3
    tsdf = bolck_tsdfTransform(point, shape, trans, quat)
    tsdf=tsdf.tolist()
    IOU=[]
    for bs in range(shape.size(0)):
        IOU.append([])
        for i in range(config.primNum):
            IOU[bs].append([])
            for j in range(config.primNum):
                if i==j:
                    IOU[bs][i].append(1)
                else:
                    temp=0
                    for spn in range(150):
                        if tsdf[bs][j][150*i+spn] == 0:    #point sample on i tsdf on j    
                            temp += 1./150 
                    #print(temp)
                    IOU[bs][i].append(temp)
    return IOU
def NMS_all(shape,point,volume,batchCP,over_count):
    # bs x np x sp x 3
    pointindex = point.add(0.5).mul(32.).int().clamp(min=0, max=31).detach().tolist()
    list_cp = batchCP.detach().tolist()
    orderList = []
    IOUList =[]
    v_voxel = (1/config.gridsize)**3
    for i in range(len(pointindex)):
        IOUList.append([])
        orderList.append([])
        for j in range(len(pointindex[i])):
            IOUList[i].append(0)
            orderList[i].append(0)
    for bs in range(len(pointindex)):
        for np in range(len(pointindex[bs])):
            for s in range(len(pointindex[bs][np])):
                x = pointindex[bs][np][s][0]
                y = pointindex[bs][np][s][1]
                z = pointindex[bs][np][s][2]
                if volume[bs][0][x][y][z] == 1:
                    IOUList[bs][np] += 1/150
    for bs in range(len(pointindex)):
        for np in range(config.primNum):
            x,y,z = shape[bs][np]
            v_prim = 8*x*y*z
            orderList[bs][np] =v_prim*IOUList[bs][np]
#    for bs in range(len(pointindex)):
#        for np in range(config.primNum):
#            voxel_prim=numpy.zeros((32,32,32))# 32 x 32 x 32
#            volume_intersect = 0
#            volume_non_intersect = 0
#            x,y,z = shape[bs][np]
#            v_prim = 8*x*y*z
#            if over_count[bs][np][0] == 1:
#                volume_intersect += v_prim * IOUList[bs][np]
#                volume_non_intersect += v_prim * (1-IOUList[bs][np])
#                for s in range(150):
#                    a = pointindex[bs][np][s][0]
#                    b = pointindex[bs][np][s][1]
#                    c = pointindex[bs][np][s][2]
#                    if voxel_prim[a][b][c] == 0:
#                        voxel_prim[a][b][c] = 1
#            for i in range(config.gridsize):
#                for j in range(config.gridsize):
#                    for k in range(config.gridsize):
#                        if volume[bs][0][i][j][k] == 1 and voxel_prim[i][j][k] == 0:
#                            volume_non_intersect += v_voxel
#        orderList[bs][np] = volume_intersect/(volume_intersect+volume_non_intersect+1e-10)
    return IOUList,orderList
    
    
    
def NMS_fin(point, shape, trans, quat,volume,IOU_all,over_count):
    #point= point.view(point.size(0), point.size(1) * point.size(2), point.size(3))
    pointindex = point.add(0.5).mul(32.).int().clamp(min=0, max=31).detach().tolist()
    v_voxel = (1/config.gridsize)**3
    IOU_fin = []
    for bs in range(len(shape)):
        voxel_prim=numpy.zeros((32,32,32))# 32 x 32 x 32
        volume_intersect = 0
        volume_non_intersect = 0
        for np in range(config.primNum):
            x,y,z = shape[bs][np]
            v_prim = 8*x*y*z
            if over_count[bs][np][0] == 1:
                volume_intersect += v_prim * IOU_all[bs][np]
                volume_non_intersect += v_prim * (1-IOU_all[bs][np])
                for s in range(150):
                    a = pointindex[bs][np][s][0]
                    b = pointindex[bs][np][s][1]
                    c = pointindex[bs][np][s][2]
                    if voxel_prim[a][b][c] == 0:
                        voxel_prim[a][b][c] = 1
        for i in range(config.gridsize):
            for j in range(config.gridsize):
                for k in range(config.gridsize):
                    if voxel_prim[i][j][k] == 1 and  volume[bs][0][i][j][k] == 0 :
                        volume_non_intersect += v_voxel
        #print(volume_intersect/(volume_intersect+volume_non_intersect+1e-10))
        IOU_fin.append(volume_intersect/(volume_intersect+volume_non_intersect+1e-10))
    return IOU_fin



def evaluation(shape,trans,quat,volume,inUse,batchCP):
    block_sample = sp.SamplePoint(bs=config.batchSize, np=config.primNum, sampleNum=150,ifSurface=False)
    point = block_sample(shape, trans, quat)
    shapelist =shape.tolist()
    IOU_sym = NMS(point,shape,trans,quat)  #IOU between prim
    IOU_all,orderList = NMS_all(shapelist,point,volume,batchCP,inUse) #IOU of hole model
    inUseNew = []
    
    for bs in range(shape.size(0)):
        inUseNew.append([])
        for np in range(config.primNum):
            IOU_all[bs][np] = IOU_all[bs][np] * inUse[bs][np][0]

    for bs in range(shape.size(0)):
        IOU_one = numpy.array(orderList[bs])
        order = numpy.flipud(numpy.argsort(IOU_one)) #big to small
        #print(order)
        over_count = [1 for _ in range(config.primNum)]
        for np in range(config.primNum):
            if inUse[bs][np][0] == 0:
                over_count[np] =0
        for np in range(config.primNum):
            prim = order[np]
            #print(IOU_one[prim])
            if IOU_one[prim] != 0:
                for snp in range(np+1,config.primNum):
                    prim_s = order[snp]
                    
                    IOU_ba = IOU_sym[bs][prim_s][prim]
                    IOU_ab = IOU_sym[bs][prim][prim_s]
                    if (IOU_ba >= threshold or IOU_ab >= threshold):
                        over_count[prim_s] = 0   #not unused or repeat
                        #print("bs"+str(bs)+" prim "+str(prim_s)+" abandoned" + str(IOU_ba) +' '+str(IOU_ab))
        #print(over_count)
        inUseNew[bs].append(over_count) #save new inUse
        #get full IOU
    IOU_fin = NMS_fin(point, shape.abs().tolist(), trans, quat,volume,IOU_all,inUse)

    return IOU_fin,inUseNew











