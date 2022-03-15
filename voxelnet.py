import torch.nn as nn
import torch.nn.functional as F
import torch
from config import config
from torch.nn import init


# conv3d + bn + relu
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True, activation=True,if_dropout=False):
        super(Conv3d, self).__init__()
        self.activation =activation
        self.batch_norm =batch_norm
        self.if_dropout=if_dropout
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(k,k,k), stride=s, padding=p)
        if self.batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.if_dropout:
            block_dropout = nn.Dropout(0.5)
            x = block_dropout(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.leaky_relu(x,0.2, inplace= True)
        else:
            return x
# max pooling Network
class MaxPooling3d(nn.Module):
    def __init__(self, k, s):
        super(MaxPooling3d, self).__init__()
        self.mp = nn.MaxPool3d((k,k,k), stride=(s,s,s))

    def forward(self, x):
        x = self.mp(x)
        return x

# # Fully Connected Network
# class FCN(nn.Module):
#
#     def __init__(self,cin,cout):
#         super(FCN, self).__init__()
#         self.cout = cout
#         self.linear = nn.Linear(cin, cout)
#         self.bn = nn.BatchNorm1d(cout)
#
#     def forward(self,x):
#         # KK is the stacked k across batch
#         kk, t, _ = x.shape
#         x = self.linear(x.view(kk*t,-1))
#         x = F.relu(self.bn(x))
#         return x.view(kk,t,-1)
# #moduel for
class ShapePred_org(nn.Module):
    def __init__(self,shapeLrDecay=config.shapeLrDecay,gridBound = config.gridBound,shape_feathers=config.shapefeathers):
        super(ShapePred_org, self).__init__()
        self.shapeLrDecay = shapeLrDecay
        self.gridBound = gridBound
        self.FCN = nn.Linear(in_features=64, out_features=shape_feathers)

        #bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                nn.init.constant_(m.bias, -3/self.shapeLrDecay)
    def forward(self, x):
        shape_x = self.FCN(x.view(x.size(0), -1))
        shape_x = shape_x.mul(self.shapeLrDecay)
        shape_x = shape_x.sigmoid()
        shape_x = shape_x.mul(config.shapeBound)
        #shape_x = shape_x.add(-0.5)
        shape_x = shape_x.squeeze()
        return shape_x
class transPred_org(nn.Module):
    def __init__(self, gridBound=config.gridBound,moveBond = config.moveBound):
        super(transPred_org, self).__init__()
        self.gridBound = gridBound
        self.moveBound =moveBond
        self.FCN = nn.Linear(in_features=64, out_features=3)
        # bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        trans_x = self.FCN(x.view(x.size(0), -1))
        trans_x = trans_x.mul(config.transLrDecay)
        trans_x = trans_x.tanh()
        trans_x = trans_x.mul(self.moveBound)
        trans_x = torch.squeeze(trans_x)
        return trans_x


class quatPred_org(nn.Module):
    def __init__(self,batchSize =config.batchSize):
        super(quatPred_org, self).__init__()
        self.FCN = nn.Linear(in_features=64, out_features=4)
        self.batchSize = batchSize

        # bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        bias_quat = torch.Tensor([1., 0., 0., 0.]).cuda().unsqueeze(0)
        bias_quat = bias_quat.expand(self.batchSize, -1)
        quat_x = self.FCN(x.view(x.size(0), -1)).add(bias_quat)
        quat_x = quat_x.squeeze()
        quat_x = F.normalize(quat_x,dim = 1)
        return quat_x

class confiPred_org(nn.Module):
    def __init__(self):
        super(confiPred_org, self).__init__()
        self.FCN = nn.Linear(in_features=64, out_features=1)
        # bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        confi_x = self.FCN(x.view(x.size(0), -1))
        confi_x = torch.tanh(confi_x)
        confi_x = torch.mul(confi_x, 0.5)
        confi_x = torch.add(confi_x, 0.5)
        confi_x = torch.squeeze(confi_x)
        return confi_x

class ShapePred(nn.Module):
    def __init__(self,shapeLrDecay=config.shapeLrDecay,gridBound = config.gridBound,shape_feathers=config.shapefeathers):
        super(ShapePred, self).__init__()
        self.shapeLrDecay = shapeLrDecay
        self.gridBound = gridBound
        self.FCN = nn.Linear(in_features=128+config.primNum*10, out_features=shape_feathers)
        #bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()
                #nn.init.constant_(m.bias, -3/self.shapeLrDecay)
    def forward(self, x):
        shape_x = self.FCN(x.view(x.size(0), -1))
        shape_x = shape_x.mul(self.shapeLrDecay)
        shape_x = shape_x.tanh()
        #shape_x = shape_x.add(0.023).mul(config.shapeBound) #used when base
        shape_x = shape_x.mul(config.shapeBound)
        #shape_x = shape_x.add(-0.5)
        shape_x = shape_x.squeeze()
        return shape_x



class transPred(nn.Module):
    def __init__(self, gridBound=config.gridBound,moveBond = config.moveBound):
        super(transPred, self).__init__()
        self.gridBound = gridBound
        self.moveBound =moveBond
        self.FCN = nn.Linear(in_features=128+config.primNum*10, out_features=3)
        # bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        trans_x = self.FCN(x.view(x.size(0), -1))
        trans_x = trans_x.mul(config.transLrDecay)
        trans_x = trans_x.tanh()
        trans_x = trans_x.mul(self.moveBound)
        trans_x = torch.squeeze(trans_x)
        return trans_x


class quatPred(nn.Module):
    def __init__(self,batchSize =config.batchSize):
        super(quatPred, self).__init__()
        self.FCN = nn.Linear(in_features=128+config.primNum*10, out_features=4)
        self.batchSize = batchSize

        # bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        bias_quat = torch.Tensor([1., 0., 0., 0.]).cuda().unsqueeze(0)
        bias_quat = bias_quat.expand(self.batchSize, -1)
        quat_x = self.FCN(x.view(x.size(0), -1)).mul(0.01).add(bias_quat)
        quat_x = quat_x.squeeze()
        quat_x = F.normalize(quat_x,dim = 1)
        return quat_x

class confiPred(nn.Module):
    def __init__(self):
        super(confiPred, self).__init__()
        self.FCN = nn.Linear(in_features=128+config.primNum*10, out_features=1)
        # bias and weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        confi_x = self.FCN(x.view(x.size(0), -1))
        confi_x = torch.tanh(confi_x)
        confi_x = torch.mul(confi_x, 0.5)
        confi_x = torch.add(confi_x, 0.5)
        confi_x = torch.squeeze(confi_x)
        return confi_x
class catFeature(nn.Module):
    def __init__(self):
        super(catFeature, self).__init__()
    def forward(self,x,shape,trans,quat):
        x = x.view(x.size(0),-1)
        shape = shape.view(x.size(0),-1).contiguous()
        trans = trans.view(x.size(0),-1).contiguous()
        quat = quat.view(x.size(0),-1).contiguous()
        rlt = torch.cat((x,shape,trans,quat),dim=1)
        rlt =rlt.view(rlt.size(0),rlt.size(1),1,1,1)
        return rlt
class primitivePred(nn.Module):
    def __init__(self,celldan =config.celldan,gridsize=config.gridsize,primNum=config.primNum,batchSize =config.batchSize,
                 gridBound=config.gridBound,shapeLrDecay=config.shapeLrDecay,ifShapeOrg=config.ifShapeOrg):
        super(primitivePred, self).__init__()
        # block for (confidance ,shape,trans, quat)*primNum
        #output = B  X primNum  X xx need transpose
        self.celldan = celldan
        self.gridsize = gridsize
        self.primNum =primNum
        self.batchSize = batchSize
        self.shapeLrDecay =shapeLrDecay
        self.gridBound =gridBound
        if config.ifShapeOrg:
            self.block_shape = nn.ModuleList([ShapePred_org(self.shapeLrDecay, self.gridBound) for _ in range(self.primNum)])
            self.block_trans = nn.ModuleList([transPred_org(self.gridBound) for _ in range(self.primNum)])
            self.block_quat = nn.ModuleList([quatPred_org() for _ in range(self.primNum)])
            self.block_confi = nn.ModuleList([confiPred_org() for _ in range(self.primNum)])
        else:
            self.block_shape = nn.ModuleList([ShapePred(self.shapeLrDecay, self.gridBound) for _ in range(self.primNum)])
            self.block_trans = nn.ModuleList([transPred(self.gridBound) for _ in range(self.primNum)])
            self.block_quat = nn.ModuleList([quatPred() for _ in range(self.primNum)])
            self.block_confi = nn.ModuleList([confiPred() for _ in range(self.primNum)])

    def forward(self, x):
        shape_list = []
        trans_list = []
        quat_list = []
        confi_list = []
        pos_w = torch.arange(0, 32).float().mul(1 / 32).add(-0.5).squeeze().tolist()
        #print(pos_w)
        for i in range(self.primNum):
            # d_num = 32 // self.celldan
            # x_i = i % d_num * self.celldan + self.celldan // 2
            # y_i = i // d_num % d_num * self.celldan + self.celldan // 2
            # z_i = i // d_num // d_num * self.celldan + self.celldan // 2
            # pos_i = [pos_w[x_i], pos_w[y_i], pos_w[z_i]]
            # pos = torch.Tensor(pos_i).cuda().unsqueeze(0)
            # pos = pos.expand(self.batchSize, -1)
            shape_list.append(self.block_shape[i](x))
            #trans_list.append(self.block_trans[i](x) + pos)
            trans_list.append(self.block_trans[i](x))
            quat_list.append(self.block_quat[i](x))
            confi_list.append(self.block_confi[i](x))
        shape_rlt = torch.stack(shape_list, dim=0).transpose(0, 1).contiguous().reshape(self.batchSize, self.primNum, 3).contiguous()
        trans_rlt = torch.stack(trans_list, dim=0).transpose(0, 1).contiguous().reshape(self.batchSize, self.primNum, 3).contiguous()
        quat_rlt = torch.stack(quat_list, dim=0).transpose(0, 1).contiguous().reshape(self.batchSize, self.primNum, 4).contiguous()
        confi_rlt = torch.stack(confi_list, dim=0).transpose(0, 1).contiguous().reshape(self.batchSize, self.primNum, 1).contiguous()
        return shape_rlt,trans_rlt,quat_rlt,confi_rlt

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        #conv layer
        self.block_1 =[Conv3d(1,4,3,s=1, p=1)]
        self.block_1 += [MaxPooling3d(2,2)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv3d(4, 8, 3, s=1, p=1)]
        self.block_2 += [MaxPooling3d(2, 2)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv3d(8, 16, 3, s=1, p=1)]
        self.block_3 += [MaxPooling3d(2, 2)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.block_4 = [Conv3d(16, 32, 3, s=1, p=1)]
        self.block_4 += [MaxPooling3d(2, 2)]
        self.block_4 = nn.Sequential(*self.block_4)

        self.block_5 = [Conv3d(32, 64, 3, s=1, p=1)]
        self.block_5 += [MaxPooling3d(2, 2)]
        self.block_5 = nn.Sequential(*self.block_5)
        #current channel_out=64
        #2 layer of fc
        if config.ifShapeOrg:
            self.block_fc = [Conv3d(64,64, 1, s=1, p=0)]
            self.block_fc += [Conv3d(64,64, 1, s=1, p=0)]
            self.block_fc = nn.Sequential(*self.block_fc)
        else:
            self.block_fc = [Conv3d(128+config.primNum*10, 128+config.primNum*10, 1, s=1, p=0)]
            self.block_fc += [Conv3d(128+config.primNum*10, 128+config.primNum*10, 1, s=1, p=0)]
            self.block_fc = nn.Sequential(*self.block_fc)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
        self.block_cat = catFeature()
        self.block_prim =primitivePred()




    def forward(self,x,y, shape, trans, quat):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        if config.ifShapeOrg:
            x = self.block_fc(x)
            shape_rlt, trans_rlt, quat_rlt, confi_rlt = self.block_prim(x)
        else:
            y = self.block_1(y)
            y = self.block_2(y)
            y = self.block_3(y)
            y = self.block_4(y)
            y = self.block_5(y)
            #x = self.block_fc(x)
            catxy = torch.cat((x,y), dim = 1)
            cat_all = self.block_cat(catxy,shape, trans, quat)
            #print(catxy.size())
            cat_all= self.block_fc(cat_all)
            shape_rlt, trans_rlt, quat_rlt, confi_rlt = self.block_prim(cat_all)
        return shape_rlt, trans_rlt, quat_rlt, confi_rlt

