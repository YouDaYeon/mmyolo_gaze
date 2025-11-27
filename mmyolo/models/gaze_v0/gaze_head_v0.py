import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.utils import (ConfigType, OptConfigType, OptMultiConfig)
from mmdet.structures import SampleList
import math
from torch import Tensor
from mmyolo.registry import MODELS

import matplotlib.pyplot as plt

import torch.nn.functional as F
from mmengine.structures import InstanceData

# from .prompt_encoder import PromptEncoder
# from .sam_decoder import TwoWayTransformer

from .CBAM import CBAM
from .position_map import PositionMap

@MODELS.register_module()
class GazeHeadModule_v0(BaseModule):
    """
        uni_feature를 feature fusion 모듈(self-attention)

        l_feature : yolov8 neck p5 와 position map 을 더한 feature
        g_feature : l_feature를 SwinTransformer 거친 output feature
        uni_feature : concat(l_feature , g_feature)
        
        output : xyz, x, y, z
    """
    def __init__(self,
                #  num_classes: int,
                #  featmap_strides,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._init_layers()
        

    def _init_layers(self):
        """
        Feature Fusion 모듈 부르기
        
        """
        # confidence 8 실험
        self.in_channels = [192, 384, 576]
        # self.in_channels = [256, 512, 512]         # yolov8l
        # self.in_channels = [256, 128, 128]         # bisenet(resnet) 실험
        # self.pool = nn.ModuleList()
        self.conv = nn.ModuleList()

        # self.head_pose_estimator = nn.ModuleList()

        for i in range(3):
            # self.head_pose_estimator.append(HeadPoseEstimator(in_channels=self.in_channels[i], hidden_dim=128, out_dim=3))
            # self.pool.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
            self.conv.append(nn.Sequential(
                # nn.Conv2d(in_channels=self.in_channels[i]*2, out_channels=self.in_channels[i], kernel_size=1),
                nn.Conv2d(in_channels=self.in_channels[i], out_channels=3, kernel_size=1),
                nn.Tanh()
                ))
            
        self.dwconv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, groups=3),  # DWConv
            nn.Tanh()
        )
        self.learnable_parameter = nn.Parameter(torch.randn(1, 3))  # p3,p4,p5 weights

        # neck에서 feature 하나만 사용
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.conv = nn.Conv2d(in_channels=192, out_channels=3, kernel_size=1) # only p3
        # self.conv = nn.Conv2d(in_channels=384, out_channels=3, kernel_size=1) # only p4
        # self.conv = nn.Conv2d(in_channels=576, out_channels=3, kernel_size=1) # only p5
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        input: uni_feature = concat(l_feature , g_feature)
        output: uni_feature를 feature fusion 모듈(self-attention)한 결과
        """

        # confidence 8 실험
        pred_p3p4p5 = []
        for i in range(3):
            # pool_x = self.pool[i](x[i])

            # pool_x = pool_x.view(pool_x.shape[0], -1)
            # head_pose_estimator_x = self.head_pose_estimator[i](x[i])
            # concat_x = torch.cat([pool_x, head_pose_estimator_x], dim=1)  # channel + 3
            # concat_x = concat_x.unsqueeze(-1).unsqueeze(-1)

            # pred_p3p4p5.append(self.conv[i](pool_x))
            # pred_p3p4p5.append(self.conv[i](concat_x))         # conv로 -> shape 변경.
            pred_p3p4p5.append(self.conv[i](x[i]))         # RoIAlign 사용 시 gaze head에서 이미 pool함.

        pred_p3p4p5 = [pred.view(pred.shape[0], -1) for pred in pred_p3p4p5] # (32, 3)
        pred_p3p4p5_concat = torch.stack(pred_p3p4p5, dim=2)  # (32, 3, 3) -> (B, xyz, p3p4p5)
        learnable_parameter = self.learnable_parameter.expand(pred_p3p4p5_concat.shape) # (B, xyz, p3p4p5)
        pred_xyz_multi = pred_p3p4p5_concat * learnable_parameter
        pred_xyz = self.dwconv(pred_xyz_multi).squeeze(-1)                     # (32, 3) -- DWConv

        # # neck에서 feature 하나만 사용
        # pool_x = self.pool(x)
        # pred_xyz = self.conv(pool_x).squeeze(-1).squeeze(-1)

        # # only p3 feature
        # pred_xyz = pred_p3p4p5[2]

        return pred_xyz, pred_p3p4p5
        # return pred_xyz

@MODELS.register_module()
class GazeHead_v0(BaseDenseHead):
    def __init__(self,
                 gaze_head_module: ConfigType,
                 loss_gaze: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.train_cfg = train_cfg 
        self.test_cfg = test_cfg

        self.gaze_head_module = MODELS.build(gaze_head_module)
        self.loss_gaze = MODELS.build(loss_gaze)

        ## CBAM 실험 
        self.gate_channels = [192, 384, 576]
        # self.gate_channels = [256, 512, 512]         # yolov8l
        # self.size = [40, 20, 10]
        self.cbam = nn.ModuleList()
        # self.msglam = nn.ModuleList()
        # self.criss_cross_attention = nn.ModuleList()  ### CCA 순차 적용 실험
        for i in range(3):
            self.cbam.append(CBAM(gate_channels=self.gate_channels[i], reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False))
            # self.cbam.append(CBAM(gate_channels=self.gate_channels[i], reduction_ratio=16, pool_types=['strip'], no_spatial=False))
            # self.msglam.append(MSGLAM(in_channels=self.gate_channels[i], groups=4, rounds=4))
            # self.cbam.append(JointCBAM(gate_channels=self.gate_channels[i], size=self.size[i], reduction_ratio=16, pool_types=['avg', 'max']))

        # roi_size = (7, 7)
        # self.roi_align = ROIAlignModule(output_size=roi_size)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # # self.face_emb = nn.ParameterList([nn.Parameter(torch.randn(1, dim, *roi_size), requires_grad=True) for dim in self.gate_channels])

        self.position_map = PositionMap()

        # # neck에서 feature 하나만 사용
        # self.cbam = CBAM(gate_channels=192, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False) # only p3
        # # self.cbam = CBAM(gate_channels=384, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False) # only p4
        # # self.cbam = CBAM(gate_channels=576, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False) # only p5
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.position_map = PositionMap()

    # confidence 8 실험
    def forward(self, x: Tensor, results_list: SampleList, batch_data_samples, training: bool) -> SampleList:

        # Position Map 실험
        position_map_face = []
        for i in range(3):
            face_x = (self.position_map(x[i], results_list, batch_data_samples, training))
            position_map_face.append(face_x)

        # 1. add 후 cbam
        add_x = []
        for i in range(3):
            add_x.append(x[i] + position_map_face[i])
            # add_x.append(position_map_face[i])
            # add_x.append(x[i])
            
        # cbam 적용
        cbam_x = []
        for i in range(3):
            cbam_x.append(self.pool(self.cbam[i](add_x[i])))

        # # no cbam 실험
        # cbam_x = []
        # for i in range(3):
        #     cbam_x.append(self.pool(add_x[i]))
        
        pred_xyz, pred_p3p4p5 = self.gaze_head_module(cbam_x)
        # pred_xyz = self.gaze_head_module(cbam_x)
        pred_xyz = F.normalize(pred_xyz, p=2, dim=1)
        pred_p3p4p5 = [F.normalize(pred, p=2, dim=1) for pred in pred_p3p4p5]


        # # neck에서 feature 하나만 사용
        # position_map_face = self.position_map(x, results_list, batch_data_samples, training)
        # cbam_x = self.cbam(x + position_map_face)
        # pred_xyz = self.gaze_head_module(cbam_x)
        # pred_xyz = F.normalize(pred_xyz, p=2, dim=1)

        return pred_xyz, pred_p3p4p5
        # return pred_xyz

    # # cossin 실험
    # def forward(self, x: Tensor, results_list: SampleList, batch_data_samples, training: bool) -> SampleList:

    #     bbox_embedding = []
    #     l_feature = []
    #     for i in range(len(x)):
    #         bbox_embedding.append(self.prompt_encoder[i](x[i], results_list, batch_data_samples, training))
    #         image_pe = self.prompt_encoder[i].get_dense_pe_2(i)
    #         l_feature.append(self.sam_decoder[i](x[i], image_pe, bbox_embedding[i])) # (tgt, memory)

    #     pred_angular, pred_var, angular_outputs_p3p4p5, vars_p3p4p5 = self.gaze_head_module(l_feature)    # [sin(yaw), cos(yaw), sin(pitch)], var

    #     return pred_angular, pred_var, angular_outputs_p3p4p5, vars_p3p4p5
    
    # confidence 8 실험
    def loss(self, x: Tensor, results_list, batch_data_samples: dict) -> SampleList:
        """
            backpropagation에서 가중치 업데이트하는데 사용되는 loss 계산
        """

        gaze_labels = batch_data_samples['gazes_labels']  # gaze gt값

        pred_xyz, pred_p3p4p5 = self(x, results_list, batch_data_samples, training = True)
        l_xyz, l_p3p4p5, l_heatmap, l_heatmap_p3p4p5 = self.loss_gaze(pred_xyz, pred_p3p4p5, gaze_labels)
        # l_xyz, l_p3p4p5 = self.loss_gaze(pred_xyz, pred_p3p4p5, gaze_labels)

        # # neck에서 feature 하나만 사용
        # pred_xyz = self(x, results_list, batch_data_samples, training = True)
        # l_xyz, l_heatmap = self.loss_gaze(pred_xyz, gaze_labels)

        return l_xyz, l_p3p4p5, l_heatmap, l_heatmap_p3p4p5
        # return l_xyz, l_p3p4p5
        # return l_xyz, l_heatmap
    
    # # cossin 실험
    # def loss(self, x: Tensor, results_list, batch_data_samples: dict) -> SampleList:
    #     """
    #         backpropagation에서 가중치 업데이트하는데 사용되는 loss 계산
    #     """

    #     gaze_labels = batch_data_samples['gazes_labels']  # (96, 3)
    #     gaze_labels = gaze_labels[::3]  # 3개씩 건너뛰면서 선택 # (32, 3)

    #     # gaze_labels를 pitch, yaw로 변환
    #     gaze_labels_yaw_pitch = self.vector_to_yaw_pitch(gaze_labels)  # (32, 2)
        
    #     gaze_sincos_vector = torch.zeros((gaze_labels_yaw_pitch.size(0), 3))
    #     # sin(Yaw)
    #     gaze_sincos_vector[:,0] = torch.sin(gaze_labels_yaw_pitch[:,0])
    #     # cos(Yaw)
    #     gaze_sincos_vector[:,1] = torch.cos(gaze_labels_yaw_pitch[:,0])
    #     # sin(Pitch)
    #     gaze_sincos_vector[:,2] = torch.sin(gaze_labels_yaw_pitch[:,1])

    #     # pitch, yaw mse 실험
    #     pred_angular, pred_var, angular_outputs_p3p4p5, vars_p3p4p5 = self(x, results_list, batch_data_samples, training = True)
    #     l_pinball, l_pinball_p3p4p5 = self.loss_gaze(pred_angular, pred_var, angular_outputs_p3p4p5, vars_p3p4p5, gaze_sincos_vector)

    #     return l_pinball, l_pinball_p3p4p5
    
    # confidence 8 실험
    def predict(self, x: Tensor, results_list, batch_data_samples) -> SampleList:

        pred_xyz, pred_p3p4p5 = self(x, results_list, batch_data_samples, training = False)
        # pred_xyz = self(x, results_list, batch_data_samples, training = False)

        final_gaze_xyz=[]
        for gaze_pred in pred_xyz:
            final_gaze_xyz.append(InstanceData(gazes = gaze_pred))

        return final_gaze_xyz
    
    # # cossin 실험
    # def predict(self, x: Tensor, results_list, batch_data_samples) -> SampleList:

    #     # pred
    #     pred_angular, pred_var, angular_outputs_p3p4p5, vars_p3p4p5 = self(x, results_list, batch_data_samples, training = False)
    #     yaw_pitch_cosine = self.get_cosinebased_yaw_pitch(pred_angular)
    #     yaw_pitch_sine = self.get_sinebased_yaw_pitch(pred_angular)

    #     pred_yaw = self.average_angle(yaw_pitch_cosine[:,0], yaw_pitch_sine[:,0]).view(-1,1)
    #     pred_pitch = self.average_angle(yaw_pitch_cosine[:, 1], yaw_pitch_sine[:, 1]).view(-1, 1)
    #     pred = torch.cat([pred_yaw, pred_pitch], dim=1)

    #     pred = self.yaw_pitch_to_vector(pred) # (32, 3)

    #     final_gaze_xyz=[]
    #     for gaze_pred in pred:
    #         final_gaze_xyz.append(InstanceData(gazes = gaze_pred))

    #     return final_gaze_xyz
    
    def loss_by_feat(self,x):
        # 이 함수 없으면 오류남
        # 주어진 feature map 과  gt를 기반으로 손실 계산
        # 이 부분에서 loss 로 넘어감
        return x
    
    def yaw_pitch_to_vector(self, x):
        '''
            라디안 값으로 들어옴.
            x[:,0]: 세타 각   (yaw, φ) 
            x[:,1]: 파이 각   (pitch, θ)

            x=r⋅cos(pitch)⋅sin(yaw)
            y=r⋅sin(pitch)
            z=r⋅cos(pitch)⋅cos(yaw)
        '''

        # x = torch.reshape(x, (-1, 2))
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