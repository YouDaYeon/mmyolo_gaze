import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS

from mmengine.dist import get_dist_info
from mmdet.models.losses.gfocal_loss import distribution_focal_loss

from .gaussian_heatmap import GaussianHeatmap
import math

@MODELS.register_module()
class GazeLoss_v0(nn.Module):
    """Gaze Arccos loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.  손실 계산에 사용하는 임계값을 설정
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.         손실을 어떻게 축소할지 정의
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.                손실에 가중치를 부여
    """

    def __init__(self, loss_weight, batch_size):
        super().__init__()
        self.loss_weight = loss_weight  # 6.0
        self.batch_size = batch_size
        # # self.gaze_dfl_w = 2
        # # self.reg_max = 32  # dfl

        self.mse_loss = nn.MSELoss(reduction='sum')
        # self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # self.mae_loss = nn.L1Loss(reduction='sum')
        # self.mae_w = 3     # mae

        # self.heatmap = GaussianHeatmap(sigma=1)
        self.heatmap = GaussianHeatmap(sigma=3)
        # self.heatmap = GaussianHeatmap(sigma=5)
        # self.heatmap = GaussianHeatmap(sigma=7)
        # self.heatmap = GaussianHeatmap(sigma=9)
        # self.heatmap = GaussianHeatmap(sigma=10)

    # confidence 8 실험
    def forward(self,
                pred_xyz,
                pred_p3p4p5,
                target):

        new_target = target[::3]  # 3개씩 건너뛰면서 선택 # (32, 3)
        # new_target = target[:]  # only face detection

        loss_xyz = self.Base_3D_cosim_acos(pred_xyz, new_target)
        loss_p3 = self.Base_3D_cosim_acos(pred_p3p4p5[0], new_target)
        loss_p4 = self.Base_3D_cosim_acos(pred_p3p4p5[1], new_target)
        loss_p5 = self.Base_3D_cosim_acos(pred_p3p4p5[2], new_target)

        pred_h = self.heatmap(size=(64, 64), gaze=pred_xyz)
        target_h = self.heatmap(size=(64, 64), gaze=new_target)

        pred_h_p3 = self.heatmap(size=(64, 64), gaze=pred_p3p4p5[0])
        pred_h_p4 = self.heatmap(size=(64, 64), gaze=pred_p3p4p5[1])
        pred_h_p5 = self.heatmap(size=(64, 64), gaze=pred_p3p4p5[2])

        loss_heatmap = self.Heatmap_loss(pred_h.squeeze(1), target_h)
        loss_heatmap_p3 = self.Heatmap_loss(pred_h_p3.squeeze(1), target_h)
        loss_heatmap_p4 = self.Heatmap_loss(pred_h_p4.squeeze(1), target_h)
        loss_heatmap_p5 = self.Heatmap_loss(pred_h_p5.squeeze(1), target_h)

        loss_p3p4p5 = loss_p3 + loss_p4 + loss_p5
        loss_heatmap_p3p4p5 = loss_heatmap_p3 + loss_heatmap_p4 + loss_heatmap_p5


        # # pitch loss 추가
        # pred_yaw_pitch = self.vector_to_yaw_pitch(pred_xyz)
        # target_yaw_pitch = self.vector_to_yaw_pitch(new_target)
        # # loss_yaw = self.mse_loss(pred_yaw_pitch[:,0], target_yaw_pitch[:,0])
        # loss_pitch = self.mse_loss(pred_yaw_pitch[:,1], target_yaw_pitch[:,1])

        # loss_pitch = loss_pitch

        return loss_xyz, loss_p3p4p5, loss_heatmap, loss_heatmap_p3p4p5
        # return loss_xyz, loss_p3p4p5

    # # neck에서 feature 하나만 사용
    # def forward(self,
    #             pred_xyz,
    #             target):

    #     new_target = target[::3]  # 3개씩 건너뛰면서 선택 # (32, 3)

    #     loss_xyz = self.Base_3D_cosim_acos(pred_xyz, new_target)

    #     pred_h = self.heatmap(size=(64, 64), gaze=pred_xyz)
    #     target_h = self.heatmap(size=(64, 64), gaze=new_target)

    #     loss_heatmap = self.Heatmap_loss(pred_h.squeeze(1), target_h)

    #     return loss_xyz, loss_heatmap
    
    def get_cosinebased_yaw_pitch(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor with two columns being yaw and pitch respectively. For yaw, it uses cos(yaw)'s value along with
        sin(yaw)'s sign.
        Args:
            input_: 1st column is sin(yaw), 2nd Column is cos(yaw), 3rd Column is sin(pitch)
        """

        yaw_pitch_cosine = torch.zeros((input_.shape[0], 2))
        yaw_pitch_cosine[:, 1] = torch.asin(input_[:, 2])

        yaw = torch.acos(input_[:, 1])
        right = (input_[:, 0] < 0.)
        yaw[right] = -1 * yaw[right]

        yaw_pitch_cosine[:, 0] = yaw
        return yaw_pitch_cosine


    def get_sinebased_yaw_pitch(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor with two columns being yaw and pitch respectively. For yaw, it uses sin(yaw)'s value along with
        cos(yaw)'s sign.
        Args:
            input_: 1st column is sin(yaw), 2nd Column is cos(yaw), 3rd Column is sin(pitch)
        """

        yaw_pitch_sine = torch.zeros((input_.shape[0], 2))
        yaw_pitch_sine[:, 1] = torch.asin(input_[:, 2])

        sin_based_yaw = torch.asin(input_[:, 0])
        back = (input_[:, 1] < 0)
        pos_yaw = sin_based_yaw >= 0

        pos_back = pos_yaw & back
        neg_back = (~pos_yaw) & back

        sin_based_yaw[pos_back] = math.pi - sin_based_yaw[pos_back]
        sin_based_yaw[neg_back] = -math.pi - sin_based_yaw[neg_back]

        yaw_pitch_sine[:, 0] = sin_based_yaw
        return yaw_pitch_sine
    
    def average_angle(self, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angle1: one dimensional tensor in radians
            angle2: one dimensional tensor in radians
        """
        c1 = torch.abs(angle1 - 2 * math.pi - angle2).view(-1, 1)
        c2 = torch.abs(angle1 - angle2).view(-1, 1)
        c3 = torch.abs(angle1 + 2 * math.pi - angle2).view(-1, 1)
        c123 = torch.cat([c1, c2, c3], dim=1)
        factor = torch.argmin(c123, dim=1) - 1

        angle1 = angle1 + factor * 2 * math.pi

        avg_angle = (angle1 + angle2) / 2

        avg_angle[avg_angle > math.pi] = avg_angle[avg_angle > math.pi] - 2 * math.pi
        avg_angle[avg_angle < -math.pi] = avg_angle[avg_angle < -math.pi] + 2 * math.pi
        return avg_angle

    def pinball_loss(self, pred, var, target):
        q1 = 0.1
        p9 = 1-q1
        q_10 = target-(pred-var)
        q_90 = target-(pred+var)

        loss_10 = torch.max(q1*q_10, (q1-1)*q_10)
        loss_90 = torch.max(p9*q_90, (p9-1)*q_90)

        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90

    def yaw_pitch_to_vector(self, x):
        '''
            x[:,1]: 파이 각
            x[:,0]: 세타 각
        '''
        x = torch.reshape(x, (-1, 2))
        output = torch.zeros((x.shape[0], 3))
        output[:,2] = - torch.cos(x[:,1]) * torch.cos(x[:,0])  # z값 계산
        output[:,0] = torch.cos(x[:,1]) * torch.sin(x[:,0])    # x값 계산
        output[:,1] = torch.sin(x[:,1])                        # y값 계산
        return output

    def vector_to_yaw_pitch(self, x):
        x = torch.reshape(x, (-1, 3))
        x = x / torch.norm(x, dim=1).reshape(-1, 1)
        output = torch.zeros((x.shape[0], 2))
        output[:,0] = torch.atan2(x[:,0], - x[:,2])
        output[:,1] = torch.asin(x[:,1])
        return output

    def Base_3D_cosim_acos(self,new_pred,new_target):

        sim = F.cosine_similarity(new_pred, new_target, dim=-1, eps=1e-6)      # 예측값과 실제값 사이의 (벡터간의 방향 차이를 )코사인 유사도를 계산. 코사인 유사도는 두 벡터가 얼마나 유사한지를 측정하며, 값은 -1에서 1 사이

        sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)                         # 유사도가 계산된 후, hardtanh 함수로 유사도를 -1에서 1 사이로 클리핑. 부동소수점 오차로 인해 acos 함수가 잘못된 값으로 처리되지 않도록 하기 위함
                                                                            # sim shape = torch.Size([32])
        loss_angle = torch.acos(sim)  # 각도 계산. 코사인 값에서 라디안 단위로 각도를 반환, 즉 loss_angle = 라디안값 , 값의 범위는 [0, π]
        
        sum_loss = loss_angle.sum()
        # mean_loss = loss_angle.mean()
        
        # _, world_size = get_dist_info()

        # return self.loss_weight * mean_loss, sim 
        return sum_loss * self.loss_weight
    
    def Heatmap_loss(self, pred, target):
        """
        pred: (B, 64, 64, 1)
        target: (B, 64, 64, 1)
        """

        # # 픽셀 단위 BCE Loss 계산
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')  # (B, 64, 64, 1)
        # 각 샘플의 평균
        loss_per_sample = torch.mean(bce_loss, dim=(1, 2))  # (B,)
        # 최종 손실 계산
        heatmap_loss = loss_per_sample.sum()  # 전체 합산

        # # 픽셀 단위 MSE Loss 계산
        # mse_loss = F.mse_loss(pred, target, reduction='none')  # (B, 64, 64, 1)
        # loss_per_sample = torch.sum(mse_loss, dim=(1, 2))
        # heatmap_loss = loss_per_sample.mean()

        return heatmap_loss * self.loss_weight
    