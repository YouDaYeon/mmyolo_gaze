import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.utils import (ConfigType, OptConfigType, OptMultiConfig)
from mmdet.structures import SampleList
import math
from torch import Tensor
from mmyolo.registry import MODELS



@MODELS.register_module()
class FrontBackClassifierHead_v0(BaseDenseHead):
    def __init__(self,
                 loss_front_back: ConfigType,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.loss_front_back = MODELS.build(loss_front_back)

        self.in_channels = [192, 384, 576]

        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) → (B, C, 1, 1)
        self.fc = nn.ModuleList()
        for i in range(3):
            self.fc.append(nn.Sequential(
                nn.Linear(self.in_channels[i], self.in_channels[i]//2),
                nn.ReLU(inplace=True),
                # nn.Linear(self.in_channels[i]//2, 1) # bce loss
                nn.Linear(self.in_channels[i]//2, 6)   # ce loss
            ))
    def forward(self, x: Tensor) -> SampleList:

        logits = []
        for i in range(3):
            x_pool = self.pool(x[i]).flatten(1)  # (B, C, 1, 1) → (B, C)
            logits.append(self.fc[i](x_pool))    # (B, 1)

        return logits

    def loss(self, x: Tensor, batch_data_samples: dict) -> SampleList:

        # # a. face bbox 기준 0 또는 1로 라벨링
        # bbox_labels = batch_data_samples['bboxes_labels']
        # face_mask = bbox_labels[:, 1] == 0
        # face_bbox_labels = bbox_labels[face_mask, 2:]

        # mask = torch.sum(face_bbox_labels, dim=1) == 0
        # target_front_back = mask.float()  # True → 1.0, False → 0.0

        # pred_front_back = self(x)
        # l_front_back = self.loss_front_back(pred_front_back, target_front_back.unsqueeze(-1))

        # # b. gaze label 기준 0 또는 1로 라벨링
        # gaze_labels = batch_data_samples['gazes_labels']
        # gaze_labels_y = gaze_labels[:, 2][::3]
        # mask = gaze_labels_y < 0
        # target_front_back = mask.float()  # True → 1.0, False → 0.0

        # pred_front_back = self(x)
        # l_front_back = self.loss_front_back(pred_front_back, target_front_back.unsqueeze(-1))


        # # c. gaze label 기준 정면 20도 / 180도 / 후면 180도 라벨링
        # gaze_labels = batch_data_samples['gazes_labels']
        # gaze_labels = gaze_labels[::3]
        # check_angular = self.check_angular(gaze_labels)
        # target_front_back = torch.zeros_like(check_angular, dtype=torch.long) # check_angular shape = (B,)
        
        # target_front_back[check_angular <= 20] = 0
        # target_front_back[(check_angular > 20) & (check_angular <= 90)] = 1  # 측면
        # target_front_back[(check_angular > 90) & (check_angular <= 160)] = 2  # 후면
        # target_front_back[check_angular > 160] = 3  # 후면 

        # pred_front_back = self(x)
        # l_front_back = self.loss_front_back(pred_front_back, target_front_back)

        # d. yaw 기준 정면 -90도 / 90도 / 180도 /-180도 라벨링
        gaze_labels = batch_data_samples['gazes_labels']
        gaze_labels = gaze_labels[::3]
        yaw = self.vector_to_yaw_pitch(gaze_labels) # (32)
        target_front_back = torch.zeros_like(yaw, dtype=torch.long) # yaw_pitch shape = (B,)
        
        # target_front_back[(yaw >= -90) & (yaw < 0)] = 0
        # target_front_back[(yaw >= 0) & (yaw < 90)] = 1
        # target_front_back[(yaw >= 90) & (yaw < 180)] = 2
        # target_front_back[(yaw >= -180) & (yaw < -90)] = 3

        # d-2
        target_front_back[(yaw >= 90) & (yaw < 160)] = 0
        target_front_back[(yaw >= 20) & (yaw < 90)] = 1
        target_front_back[(yaw >= -20) & (yaw < 20)] = 2
        target_front_back[(yaw >= -90) & (yaw < -20)] = 3
        target_front_back[(yaw >= -160) & (yaw < -90)] = 4
        target_front_back[(yaw >= -180) & (yaw < -160) & (yaw >= 160) & (yaw <= 180)] = 5

        pred_front_back = self(x)
        l_front_back = self.loss_front_back(pred_front_back, target_front_back)

        return l_front_back

    def predict(self, x) -> SampleList:
        pass
    
    def loss_by_feat(self,x):
        # 이 함수 없으면 오류남
        return x

    # def check_angular(self, gaze_labels):
    #     device = gaze_labels.device

    #     front_gaze = torch.tensor([0.,0.,-1.]).view(1,1,3).to(device)
    #     front_gaze = front_gaze.expand(gaze_labels.shape[0],-1,-1)
    #     check_gt = gaze_labels.view(-1,3,1)
    #     output_dot = torch.bmm(front_gaze,check_gt)
    #     output_dot = output_dot.view(-1)
    #     check_angular = torch.acos(output_dot) * 180 / math.pi

    #     return check_angular

    def vector_to_yaw_pitch(self, gaze_labels):
        """Convert unit vector to yaw and pitch angles
        Args:
            gaze_labels: tensor of shape (32, 3) containing unit vectors
        Returns:
            yaw_pitch: tensor of shape (32, 2) containing yaw and pitch angles in degrees
        """
        yaw_pitch = torch.zeros((gaze_labels.shape[0], 2), device=gaze_labels.device)
        yaw_pitch[:,0] = torch.atan2(gaze_labels[:,0], -gaze_labels[:,2]) * 180 / math.pi  # yaw in degrees
        yaw_pitch[:,1] = torch.asin(gaze_labels[:,1]) * 180 / math.pi  # pitch in degrees
        return yaw_pitch[:,0]  # yaw 값만 반환