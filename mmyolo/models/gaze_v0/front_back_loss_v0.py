import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS

@MODELS.register_module()
class FrontBackLoss_v0(nn.Module):
    def __init__(self, loss_weight, batch_size):
        super().__init__()
        self.loss_weight = loss_weight
        self.batch_size = batch_size

        # self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    # def forward(self, pred_front_back, target_front_back):

    #     loss_p3 = self.bce_loss(pred_front_back[0], target_front_back)
    #     loss_p4 = self.bce_loss(pred_front_back[1], target_front_back)
    #     loss_p5 = self.bce_loss(pred_front_back[2], target_front_back)

    #     loss = loss_p3 + loss_p4 + loss_p5

    #     return loss * self.loss_weight

    def forward(self, pred_front_back, target_front_back):

        loss_p3 = self.ce_loss(pred_front_back[0], target_front_back)
        loss_p4 = self.ce_loss(pred_front_back[1], target_front_back)
        loss_p5 = self.ce_loss(pred_front_back[2], target_front_back)

        loss = loss_p3 + loss_p4 + loss_p5

        return loss * 1