import transformer as tf
from config import config
import numpy
import scipy as sp
import torch
from scipy.optimize import leastsq
from scipy.optimize import fsolve,root
intersection_index= [[0,3,4],[0,3,5],[0,2,4],[0,2,5],
                    [1,3,4],[1,3,5],[1,2,4],[1,2,5]] #checked
intersection_init =[[-1,1,-1],[-1,1,1],[-1,-1,-1],[-1,-1,1],
              [1,1,-1 ], [1,1,1], [1,-1,-1 ], [1,-1,1]] #checked,mul shape when used
intersection_index_forface = [[2,3,0,1],[6,7,4,5],
                              [2,3,6,7],[0,1,4,5],
                              [2,0,6,4],[3,1,7,5]]   #checked

intersection_line_index=[[0,3],[0,2],[0,4],[0,5],
                         [1,3],[1,2],[1,4],[1,5],
                         [3,4],[3,5],[2,4],[2,5]]     #checked

intersection_line_init=[[-1,1,0],[-1,-1,0],[-1,0,-1],[-1,0,1],
                        [1,1,0],[1,-1,0],[1,0,-1],[1,0,1],
                        [0,1,-1],[0,1,1],[0,-1,-1],[0,-1,1]]  #checked,0 inplace with true value then mul shape  when used
intersection_line_index_forface = [[2,3,1,0],[6,7,5,4],
                                   [10,11,1,5],[8,9,0,4],
                                   [10,8,2,6],[11,9,3,7]]      #checked mind the sequence wehn used
###################function for surface l2norm########################
def func_re1(p,var1,var2):
    k2,k1,b = p
    return k2*var2+\
           k1*var1+\
           b

def func_re2(p, var1, var2):
    k5,k4,k3,k2,k1,b = p
    return k5*pow(var2,2)+\
           k4*var2*var1+\
           k3*pow(var1,2)+\
           k2*var2+\
           k1*var1+\
           b

def func_re3(p, var1, var2):
    k9, k8, k7, k6, k5, k4, k3, k2, k1, b = p
    return k9 * pow(var2, 3) +\
           k8 * var1 * pow(var2, 2) + \
           k7 * pow(var1, 2) * var2 + \
           k6 * pow(var1, 3) + \
           k5 * pow(var2, 2) + \
           k4 * var2 * var1 + \
           k3 * pow(var1, 2) + \
           k2 * var2 + \
           k1 * var1 + \
           b
def error1(p,var1,var2,d_ver):
    return d_ver - func_re1(p,var1,var2)
def error2(p,var1,var2,d_ver):
    return d_ver - func_re2(p,var1,var2)
def error3(p,var1,var2,d_ver):
    return d_ver - func_re3(p,var1,var2)
def l2norm(sep_point,bs,np,face,term,shape):
    listpoint = sep_point[bs][np][face] #get sample point
    if face == 0 :
        para = -shape[0]
    elif face == 1:
        para = shape[0]
    elif face == 2 :
        para = -shape[1]
    elif face == 3:
        para = shape[1]
    elif face == 4 :
        para = -shape[2]
    elif face == 5:
        para = shape[2]
    if (len(listpoint) <=50):
        return 1e10,[0,0,para]
    
    if face == 0 or face == 1:  #x = f(y,z)
        d_var = numpy.array(listpoint)[:,0]
        i_var1 = numpy.array(listpoint)[:,1]
        i_var2 = numpy.array(listpoint)[:,2]
    elif face == 2 or face == 3: #y = f(x,z)
        d_var = numpy.array(listpoint)[:,1]
        i_var1 = numpy.array(listpoint)[:,0]
        i_var2 = numpy.array(listpoint)[:,2]
    elif face == 4 or face == 5:  #z = f(x,y)
        d_var = numpy.array(listpoint)[:,2]
        i_var1 = numpy.array(listpoint)[:,0]
        i_var2 = numpy.array(listpoint)[:,1]
    #init para
    if term == 1:
        p = [0, 0, para]
    elif term == 2:
        p = [0,0,0,0,0,para]
    elif term == 3:
        p = [0,0,0,0,0,0,0,0,0,para]

    if term == 1:
        Para = leastsq(error1, p, args=(d_var, i_var1,i_var2))
    elif term == 2:
        Para = leastsq(error2, p, args=(d_var, i_var1, i_var2))
    elif term == 3:
        Para = leastsq(error3, p, args=(d_var, i_var1, i_var2))
    error = 0
    for point in range(len(listpoint)):
        if term == 1:
            temp = error1(Para[0],i_var1[point],i_var2[point],d_var[point])
        elif term == 2:
            temp = error2(Para[0],i_var1[point],i_var2[point],d_var[point])
        elif term == 3:
            temp = error3(Para[0],i_var1[point],i_var2[point],d_var[point])
        error +=abs(temp)
    error = error / len(listpoint)
    
    return error,Para[0] #error and para
###################function for border and sample########################


def getSamplePoint(para,f,i,j,shape):
    
    var1= i/3.5*2 -2
    var2 =j/3.5*2 -2
    if f == 0 or f == 1:  # x = f(y,z)
        var1,var2 = var1 *shape[1],var2 *shape[2]
    elif f == 2 or f == 3:  # y = f(x,z)
        var1,var2 = var1 *shape[0],var2*shape[2]
    elif f == 4 or f == 5:  # z = f(x,y)
        var1,var2 = var1 *shape[0],var2 *shape[1]
    if len(para) == 3:
        ans = -error1(para,var1,var2,0)
    elif len(para) == 6:
        ans = -error2(para, var1, var2, 0)
    else:
        ans = -error3(para,var1,var2,0)
    if f == 0 or f == 1:  # x = f(y,z)
        return [ans,var1,var2]
    elif f == 2 or f == 3:  # y = f(x,z)
        return [var1,ans,var2]
    elif f == 4 or f == 5:  # z = f(x,y)
        return [var1,var2,ans]
def getEquation(para,f):
    if len(para) == 3:
        k2, k1, b = para
        k9, k8, k7, k6, k5, k4, k3 =[0,0,0,0,0,0,0]
    elif len(para) == 6:
        k5, k4, k3, k2, k1, b = para
        k9, k8, k7, k6 = [0,0,0,0]
    else:
        k9, k8, k7, k6, k5, k4, k3, k2, k1, b = para
    if f == 0 or f == 1:
        def func_temp(i):
            d_var,var1,var2 = i
            return k9 * pow(var2, 3) +\
           k8 * var1 * pow(var2, 2) + \
           k7 * pow(var1, 2) * var2 + \
           k6 * pow(var1, 3) + \
           k5 * pow(var2, 2) + \
           k4 * var2 * var1 + \
           k3 * pow(var1, 2) + \
           k2 * var2 + \
           k1 * var1 + \
           b -d_var
    elif f == 2 or f==3:
        def func_temp(i):
            var1,d_var,var2 = i
            return k9 * pow(var2, 3) +\
           k8 * var1 * pow(var2, 2) + \
           k7 * pow(var1, 2) * var2 + \
           k6 * pow(var1, 3) + \
           k5 * pow(var2, 2) + \
           k4 * var2 * var1 + \
           k3 * pow(var1, 2) + \
           k2 * var2 + \
           k1 * var1 + \
           b -d_var
    elif f == 4 or f==5:
        def func_temp(i):
            var1,var2,d_var = i
            return k9 * pow(var2, 3) +\
           k8 * var1 * pow(var2, 2) + \
           k7 * pow(var1, 2) * var2 + \
           k6 * pow(var1, 3) + \
           k5 * pow(var2, 2) + \
           k4 * var2 * var1 + \
           k3 * pow(var1, 2) + \
           k2 * var2 + \
           k1 * var1 + \
           b -d_var
    return func_temp
def getIntersectionPoint(para1,f1,para2,f2,para3,f3,init): 
    fun1 = getEquation(para1 , f1)
    fun2 = getEquation(para2, f2)
    fun3 = getEquation(para3, f3)
    def f_all(i):
        return [fun1(i),fun2(i),fun3(i)]
    r = fsolve(f_all, init)   #init pos can change
    return r.tolist()
def getIntersectionPointAll(para,shape):
    points =[]
    for i in range(len(intersection_index)):
        ind1,ind2,ind3= intersection_index[i][0],intersection_index[i][1],intersection_index[i][2]
        init = [shape[0]*intersection_init[i][0],shape[1]*intersection_init[i][1],shape[2]*intersection_init[i][2]]
        points.append(getIntersectionPoint(para[ind1],ind1,para[ind2],ind2,para[ind3],ind3,init))
    return points
def getIntersectionLine(para1, f1, para2, f2,para_one,f3,init):
    fun1 = getEquation(para1, f1)
    fun2 = getEquation(para2, f2)
    fun3 = getEquation([0,0,para_one], f3)
    def f_all(i):
        return [fun1(i), fun2(i), fun3(i)]
    r = fsolve(f_all, init)  # init pos can change
    return r.tolist()
def getIntersectionLineAll(para,shape):
    lines = []
    for i in range(len(intersection_line_index)):
        lines.append([])
        for j in range(1,7):
            ind1 = intersection_line_index[i][0]
            ind2 = intersection_line_index[i][1]
            if (ind1 != 0 and ind1 !=1) and (ind2 != 0 and ind2 !=1):
                ind3 = 0
            elif (ind1 != 2 and ind1 !=3) and (ind2 != 2 and ind2 !=3):
                ind3 = 2
            else:
                ind3 = 4
            init = intersection_line_init[i]
            for w in range(3):
                if init[w]==0:
                    init[w]=j/3.5-1
            init=[shape[0]*init[0],shape[1]*init[1],shape[2]*init[2]]
#            print(para[ind1], ind1, para[ind2], ind2, (j/3.5-1.)*shape[ind3//2],ind3 ,init)
#            print(getIntersectionLine(para[ind1], ind1, para[ind2], ind2, (j/3.5-1.)*shape[ind3//2],ind3 ,init))
            lines[i].append(getIntersectionLine(para[ind1], ind1, para[ind2], ind2, (j/3.5-1.)*shape[ind3//2],ind3 ,init))
    return lines
def beautify(samplepoint,shape,trans,quat,inUse,namelist):
    # seprate point into different part
    # sent into init postion
    # seprate point into different face
    shape=shape.abs()
    shape_list = shape.tolist()
    Block_partIndex = tf.partIndex()
    samplepoint = samplepoint.unsqueeze(1).expand(-1, config.primNum, -1, -1).cuda()
    sep_point = Block_partIndex(samplepoint,shape,trans,quat,inUse) #return rigid point
    #bs,np,face, error para
    para_all =[]
    for bs in range(shape.size(0)):
        para_all.append([])
        for np in range(config.primNum):
            para_all[bs].append([])
            for f in range(6):
                error = 0
                para=[]
                for term in range(1,4):
                    error_t,para_t = l2norm(sep_point, bs, np, f, term,shape_list[bs][np])
                    #print(error_t,para_t)
                    #print(term,para_t)
                    if term == 1:
                        error = error_t
                        para = para_t
                        if error == 1e10:
                            break
                    else:
                        print(error,error_t)
                        print(term,para_t)
                        print(para)
                        if error > error_t*1.1:
                            error = error_t
                            para = para_t
                        else:
                            break
                para_all[bs][np].append(para)
    # get Parametric surface (get intersert line and point)
    # bs x np x f x 8 x 8 x 3
    ver_all = []
    for bs in range(shape.size(0)):
        ver_all.append([])
        for np in range(config.primNum):
            ver_all[bs].append([])
            #intersection_point=getIntersectionPointAll(para_all[bs][np],shape_list[bs][np])#get intersection 8x3
            #intersection_Line=getIntersectionLineAll(para_all[bs][np],shape_list[bs][np])#get intersection Line 12 x 6 x 3
            for f in range(6):
                if f == 0 or f == 1:  # x = f(y,z)
                    ind_i,ind_j =1,2
                elif f == 2 or f == 3:  # y = f(x,z)
                    ind_i,ind_j =0,2
                elif f == 4 or f == 5:  # z = f(x,y)
                    ind_i,ind_j =0,1
                ver_temp = [] #ver for 1 face 8x8x3
                for i in range(8):
                    ver_temp.append([])
                for i in range(8):
                    for j in range(8):
                        ver_temp[i].append(getSamplePoint(para_all[bs][np][f],f,i,j,shape_list[bs][np]))
                #print(ver_temp)
                ver_all[bs][np].append(ver_temp)
                #rotate and trans
    allpoint = torch.Tensor(ver_all)
    allpoint_alter = allpoint.tolist()
    for bs in range(shape.size(0)):
         for prim in range(8):
                if inUse[bs][prim] == [1]:
                    for f in range(6):
                        if f == 0:
                            delta = [2,0,0]
                        elif f ==1:
                            delta = [-2,0,0]
                        elif f ==2:
                            delta = [0,2,0]
                        elif f ==3:
                            delta = [0,-2,0]
                        elif f ==4:
                            delta = [0,0,2]
                        elif f ==5:
                            delta = [0,0,-2]
                        for i_ind in range(8):
                            for j_ind in range(8):
                                temp = allpoint[bs][prim][f][i_ind][j_ind]
                                allpoint_alter[bs][prim][f][i_ind][j_ind][0] =temp[0]+delta[0]
                                allpoint_alter[bs][prim][f][i_ind][j_ind][1] =temp[1]+delta[1]
                                allpoint_alter[bs][prim][f][i_ind][j_ind][2] =temp[2]+delta[2]
    allpoint_alter = torch.Tensor(allpoint_alter)
    allpoint_alter = allpoint_alter.view(allpoint.size(0),allpoint.size(1),allpoint.size(2)*allpoint.size(3)*allpoint.size(4),3)   
    allpoint = allpoint.view(allpoint.size(0),allpoint.size(1),allpoint.size(2)*allpoint.size(3)*allpoint.size(4),3)
    # bs x np x (6 x 8 x 8) x 3
    block_transform = tf.Transform_nn()
    shape_one = torch.ones_like(shape)
    pointlist = block_transform(allpoint, shape_one, trans, quat)
    pointlist =pointlist.view(-1,allpoint.size(1)*allpoint.size(2),3).tolist()
    pointlist_alter = block_transform(allpoint_alter, shape_one, trans, quat)
    pointlist_alter =pointlist_alter.view(-1,allpoint.size(1)*allpoint.size(2),3).tolist()
    return pointlist,pointlist_alter
    # get surface's sample point and connect them into a hole part
    # bs x np x v x 3
    # bs x np x f x 3
