import torch
import torch.nn as nn
from config import config
from quatUtils import quatToAngle
class StepLoss(nn.Module):
    def __init__(self, stepNum= config.stepNum,if_init = True):
        super(StepLoss, self).__init__()
        self.block_loss_quat = torch.nn.L1Loss(reduction='mean')
        self.block_loss_trans = torch.nn.L1Loss(reduction='mean')
        self.block_quatToAngle = quatToAngle()
        self.stepNum = stepNum
        self.if_init = if_init
    def forward(self,shape_dlt, trans_dlt, quat_dlt):
        #step x bs x np x n
        angle_dlt = self.block_quatToAngle(quat_dlt)
        
        trans_GT = torch.zeros_like(trans_dlt)
        angle_GT = torch.zeros_like(angle_dlt)

        trans_mod = torch.ones_like(shape_dlt).mul(0.02)
        angle_mod = torch.ones_like(shape_dlt).mul(3.14/18.)
        
        

        trans_mod = (trans_dlt.abs() - trans_mod).relu()
        angle_mod = (angle_dlt.abs() - angle_mod).relu()
        loss = self.block_loss_trans(trans_mod,trans_GT)+(self.block_loss_quat(angle_mod,angle_GT))
        return loss