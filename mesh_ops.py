import os
from config import config
from data import dataLoader as dt
import torch
import numpy as np
import scipy.io
import os
from loss import samplePointMethod as sp

def writeObj(listVertics, listFace, dir, iter,strl=""):
    for i in range(len(listVertics)):
        name = "iter" + str(iter) + "n" + str(i) + str(strl) + ".obj"
        fp = open(os.path.join(dir, name), 'w')
        for nv in range(0,len(listVertics[i])):
            fp.write("v " + str(listVertics[i][nv][0]) + ' '+ str(listVertics[i][nv][1]) + ' '+ str(listVertics[i][nv][2]) + '\n')
            
        for nf in range(0,len(listFace[i])):
            fp.write("f " + str(listFace[i][nf][0]) + ' '+ str(listFace[i][nf][1]) + ' '+ str(listFace[i][nf][2]) + '\n')
        fp.close()
def writeObj_prim(listVertics, dir, iter,confi_rlt,strl=""):

    confi_rlt =confi_rlt.tolist()
    confi_list=np.arange(len(confi_rlt[0])*len(confi_rlt)).reshape(len(confi_rlt),len(confi_rlt[0]))
    confi_list_index = []
    for i in range(len(confi_rlt)):
        for j in range(len(confi_rlt[i])):
            #print(i,j)
            confi_list[i][j]=confi_rlt[i][j][0]
    for i in range(len(confi_rlt)):
        confi_list_index.append(np.argsort(confi_list[i]))
    for i in range(len(listVertics)):
        count = 0
        name = "iter" + str(iter) + "n" + str(i) + str(strl) + ".obj"
        name_mtl = "iter" + str(iter) + "n" + str(i) + str(strl) + ".mtl"
        fp = open(os.path.join(dir, name), 'w')
        fp.write("mtllib " + name_mtl + '\n')
        for nv in range(0,len(listVertics[i])):
            for j in range(0,len(listVertics[i][nv])):
                count += 1
                fp.write("v " + str(listVertics[i][nv][j][0]) + ' '+ str(listVertics[i][nv][j][1]) + ' '+ str(listVertics[i][nv][j][2]) + '\n')
        for nf in range(count//8):

            fp.write("usemtl m"+str(nf+1)+'\n')
            if confi_rlt[i][nf][0] != 0:
                for t in range(12):
                    fp.write("f " + str(config.faceset[t][0]+1+nf*8) + ' '+  str(config.faceset[t][1]+1+nf*8) + ' '+  str(config.faceset[t][2]+1+nf*8) + '\n')
        fp.close()
        fp = open(os.path.join(dir, name_mtl), 'w')

        for nf in range(count // 8):
            fp.write("newmtl m"+str(nf+1)+'\n')

            fp.write("Ka 0 0 0" + '\n')
            #if confi_list_index[i][nf] <= 15 and confi_rlt[i][nf][0] != 0:
            if confi_rlt[i][nf][0] != 0:
                fp.write("Tf "+ str(confi_rlt[i][nf][0])+' '+ str(confi_rlt[i][nf][0])+' '+ str(confi_rlt[i][nf][0])+' ' +'\n')
                fp.write('Kd 0 '+str(confi_rlt[i][nf][0])+' '+ str(1-confi_rlt[i][nf][0])+' ' + '\n')
        fp.close()
def saveObj(listPrimV, iter, ListGT,confi_rlt,saveDir = config.meshDir):
    #list size: 2 x np x 3
    # mkdir for current net
    if(not os.path.exists(os.path.join(saveDir,config.netName))):
        os.mkdir(os.path.join(saveDir, config.netName))

    # load vectices and face, output obj
    for i in range(2):
        listVertics, listFace = dt.loadVFData(ListGT)
        writeObj(listVertics, listFace, os.path.join(saveDir, config.netName), iter,"_gt")
    # calculate prim, output obj
    for i in range(2):
        writeObj_prim([listPrimV[0],listPrimV[1]], os.path.join(saveDir, config.netName), iter,confi_rlt,"_prim")

def writeObjPrimLoop(shape_rlt, trans_rlt, quat_rlt, confi_rlt,listVertics,volume,Name,dir,loopNum,inUse):
    (name, _) = os.path.splitext(os.path.basename(Name))
    name_mat = name + 'loop' + str(loopNum) + ".mat"
    # for i in range(len(listVertics)):
    count = 0

    name_obj = name + 'loop' + str(loopNum) + ".obj"
    name_mtl = name + 'loop' + str(loopNum) + ".mtl"
    #print(Name)
    fp = open(os.path.join(dir, name_obj), 'w')
    for nv in range(0,len(listVertics)):
        count += 1
        for j in range(0,len(listVertics[nv])):
            fp.write("v " + str(listVertics[nv][j][0]) + ' '+ str(listVertics[nv][j][1]) + ' '+ str(listVertics[nv][j][2]) + '\n')
    for nf in range(count):
        if inUse[nf][0] > 0.:
            fp.write("usemtl m" + str(nf) + '\n')
            for t in range(12):
                fp.write("f " + str(config.faceset[t][0]+1+nf*8) + ' '+  str(config.faceset[t][1]+1+nf*8) + ' '+  str(config.faceset[t][2]+1+nf*8) + '\n')
    fp.close()
    fp = open(os.path.join(dir, name_mtl), 'w')
    fp.write("newmtl m0"+ '\n')
    fp.write("Kd 0.5843137253 0.815686274 0.8705882353"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.write("newmtl m1"+ '\n')
    fp.write("Kd 0.8705882353 0.8196078431 0.9568627451"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.write("newmtl m2"+ '\n')
    fp.write("Kd 0.6 0.737254 0.8862745098"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.write("newmtl m3"+ '\n')
    fp.write("Kd 0.619607843 0.66666666 0.8313725"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.write("newmtl m4"+ '\n')
    fp.write("Kd 0.36470588 0.7333333 0.7725490196"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.write("newmtl m5"+ '\n')
    fp.write("Kd 0.286274509 0.643137254 0.858823529"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.write("newmtl m6"+ '\n')
    fp.write("Kd 0.3411764706 0.588235294 0.8"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.write("newmtl m7"+ '\n')
    fp.write("Kd 0.380392156 0.7333333333 0.631372549"+ '\n')
    fp.write("Ka 0 0 0"+ '\n')
    fp.close()
    #getting point
    #print(os.path.join(dir, name_mat))
    scipy.io.savemat(os.path.join(dir, name_mat), {'volume':volume.tolist(),'shape_rlt': shape_rlt, 'trans_rlt': trans_rlt,'quat_rlt':quat_rlt,'confi_rlt':confi_rlt})
    #shape_rlt, trans_rlt, quat_rlt, confi_rlt


def saveStepData(shape_rlt, trans_rlt, quat_rlt,listName,index_begin,loopNum, confi_rlt,listPrimV,inUse,saveDir = config.matDir,if_mat =True):
    # list size: epoch x np x 3
    if (not os.path.exists(os.path.join(saveDir, config.netName+"loop"+str(loopNum)))):
        os.mkdir(os.path.join(saveDir, config.netName+"loop"+str(loopNum)))
    if if_mat:
        block_sp = sp.SamplePoint(bs=len(listPrimV),ifSurface=False)
        sp_rlt =block_sp(shape_rlt, trans_rlt, quat_rlt)
        pointindex = sp_rlt.add(0.5).mul(32.).int().clamp(min=0, max=31).detach().tolist()
        # epoch x np x 90 x 3
        shape_rlt= shape_rlt.tolist()
        trans_rlt= trans_rlt.tolist()
        quat_rlt = quat_rlt.tolist()
        for i in range(len(pointindex)):
            volume = np.zeros((32, 32, 32))
            for j in range(len(pointindex[i])):
                if inUse[i][j] != [0]:
                    #print(inUse[i][j])
                    for k in range(len(pointindex[i][j])):
                        if (pointindex[i][j] != 0):
                            #get index
                            x = pointindex[i][j][k][0]
                            y = pointindex[i][j][k][1]
                            z = pointindex[i][j][k][2]
                            if volume[x][y][z] < 1:
                                volume[x][y][z] += 1/90

            writeObjPrimLoop(shape_rlt[i], trans_rlt[i], quat_rlt[i], confi_rlt[i],listPrimV[i],volume,listName[(index_begin+i) %len(listName)],os.path.join(saveDir, config.netName+"loop"+str(loopNum)),loopNum,inUse[i])
    else:
        volume = np.zeros((32, 32, 32))
        for i in range(shape_rlt.size(0)):
           writeObjPrimLoop(shape_rlt[i], trans_rlt[i], quat_rlt[i], confi_rlt[i],listPrimV[i],volume,listName[(index_begin+i) %len(listName)],os.path.join(saveDir, config.netName+"loop"+str(loopNum)),loopNum,inUse[i])
def set_f(a,b,c,fp):
    fp.write("f " + str(a) + ' ' + str(c) + ' ' + str(b) + '\n') 
#    A,B,C=np.array(p_a),np.array(p_b),np.array(p_c)
#    ab = B-A
#    ac= C-A
#    normal = np.cross(ab,ac)
#    ve = normal[0]*(center[0]-p_a[0])+normal[1]*(center[1]-p_a[1])+normal[2]*(center[2]-p_a[2])
#    vf = normal[0]*(test_p[0]-p_a[0])+normal[1]*(test_p[1]-p_a[1])+normal[2]*(test_p[2]-p_a[2])
#    if ve/vf>0:
#        fp.write("f " + str(a) + ' ' + str(b) + ' ' + str(c) + '\n'
    
#    else:
#        fp.write("f " + str(a))+ ' ' + str(c) + ' ' + str(b) + '\n'
#        fp.write("f " + str(a) + ' ' + str(b) + ' ' + str(c) + '\n'))
def writeBeautyData(pointList,pointList_alter,namelist,inUse,trans):
    if (not os.path.exists(os.path.join(config.beaDir, config.netName ))):
        os.mkdir(os.path.join(config.beaDir, config.netName))
    
    # bs x np x (6 x 8 x 8) x 3
    for bs in range(len(pointList)):
        (name, _) = os.path.splitext(os.path.basename(namelist[bs]))
        if (not os.path.exists(os.path.join(config.beaDir, config.netName )+'/'+name+'/')):
            os.mkdir(os.path.join(config.beaDir, config.netName )+'/'+name+'/')
        for prim in range(8):
            bias = prim*6*8*8
            if inUse[bs][prim] == [1]:
                for f in range(6):
                    center = [(pointList[bs][bias+f*64+0][0]+pointList_alter[bs][bias+f*64+63][0])/2,(pointList[bs][bias+f*64+0][1]+pointList_alter[bs][bias+f*64+63][1])/2,(pointList[bs][bias+f*64+0][2]+pointList_alter[bs][bias+f*64+63][2])/2]
                    name_obj = os.path.join(config.beaDir, config.netName )+'/'+name+'/' +name+"prim"+str(prim)+"face"+str(f)+'.obj'
                    print(name_obj)
                    fp = open(name_obj, 'w')
                    for p_ind in range(8*8):
                        ver = f*64+p_ind
                        fp.write("v " + str(pointList[bs][bias+ver][0]) + ' ' + str(pointList[bs][bias+ver][1]) + ' ' + str(pointList[bs][bias+ver][2]) + '\n')
                    for p_ind in range(8*8):
                        ver = f*64+p_ind
                        fp.write("v " + str(pointList_alter[bs][bias+ver][0]) + ' ' + str(pointList_alter[bs][bias+ver][1]) + ' ' + str(pointList_alter[bs][bias+ver][2]) + '\n')    
                    f_bias = 64    
                    for i in range(0,7):
                        for j in range(0,7):
                            self_p = i*8+j+1
                            right = self_p+1
                            down = self_p+8
                            right_down = down +1
                            if f == 1 or f==2 or f==5:
                                set_f(self_p,right,right_down,fp)
                                set_f(self_p,right_down,down,fp)
                                set_f(self_p+f_bias,right_down+f_bias,right+f_bias,fp)
                                set_f(self_p+f_bias,down+f_bias,right_down+f_bias,fp)
                            else:
                                set_f(self_p,right_down,right,fp)
                                set_f(self_p,down,right_down,fp)
                                set_f(self_p+f_bias,right+f_bias,right_down+f_bias,fp)
                                set_f(self_p+f_bias,right_down+f_bias,down+f_bias,fp)
                # connect two face
                    for i in range(0,8):
                        if i == 0 or i==7:
                            for j in range(0,7):
                                self_p = i*8+j+1
                                right = self_p+1
                                down = self_p+64
                                right_down = right +64
                                if f == 1 or f==2 or f==5:
                                    if i==7:
                                        set_f(self_p,right,right_down,fp)
                                        set_f(self_p,right_down,down,fp)
                                    else:
                                        set_f(self_p,right_down,right,fp)
                                        set_f(self_p,down,right_down,fp)
                                else:
                                    if i==0:
                                        set_f(self_p,right,right_down,fp)
                                        set_f(self_p,right_down,down,fp)
                                    else:
                                        set_f(self_p,right_down,right,fp)
                                        set_f(self_p,down,right_down,fp)
                    for j in range(0,8):
                        if j==0 or j ==7:
                            for i in range(0,7):
                                self_p = i*8+j+1
                                right = self_p+8
                                down = self_p+64
                                right_down = right +64
                                if f == 1 or f==2 or f==5:
                                    if j==0:
                                        set_f(self_p,right,right_down,fp)
                                        set_f(self_p,right_down,down,fp)
                                    else:
                                        set_f(self_p,right_down,right,fp)
                                        set_f(self_p,down,right_down,fp)
                                else:
                                    if j==7:
                                        set_f(self_p,right,right_down,fp)
                                        set_f(self_p,right_down,down,fp)
                                    else:
                                        set_f(self_p,right_down,right,fp)
                                        set_f(self_p,down,right_down,fp)
                                   
                    fp.close()
