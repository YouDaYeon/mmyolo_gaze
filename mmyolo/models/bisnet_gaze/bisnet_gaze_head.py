import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.utils import (ConfigType, OptConfigType, OptMultiConfig)
from mmdet.structures import SampleList
import math
from torch import Tensor
from mmyolo.registry import MODELS

import torch.nn.functional as F
from mmengine.structures import InstanceData

from ..gaze_v0.prompt_encoder import PromptEncoder
from ..gaze_v0.sam_decoder import TwoWayTransformer

from ..gaze_v0.sam_decoder import Attention

from ..bisnet_gaze.common import vis_parsing_maps

import numpy as np
from PIL import Image
import cv2

@MODELS.register_module()
class BisnetGazeHeadModule_v0(BaseModule):
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

        ### reg head
        # self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(256),    
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 8 -> 4 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=2, padding=1),  # 2 -> 1
            # nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        input: uni_feature = concat(l_feature , g_feature)
        output: uni_feature를 feature fusion 모듈(self-attention)한 결과
        """

        ### reg_head
        # x = self.conv1(x)
        pred_xyz = self.conv(x).squeeze(-1).squeeze(-1)

        # return pred_xyz, pred_p3p4p5
        return pred_xyz

@MODELS.register_module()
class BisnetGazeHead_v0(BaseDenseHead):
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

        self.prompt_encoder = PromptEncoder(embed_dim=256)
        self.self_attn = Attention(embedding_dim=256, num_heads=8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    # confidence 8 실험
    def forward(self, x: Tensor, results_list: SampleList, batch_data_samples, training: bool) -> SampleList:

        ### reg head
        seg_masks = results_list.argmax(dim=1).unsqueeze(1)  # (16, 1, 64, 64)

        '''
        seg > class label
        0: 배경
        1: skin
        2: l_brow
        3: r_brow
        4: l_eye
        5: r_eye
        6: eye_g
        7: l_ear
        8: r_ear
        9: ear_r
        10: nose
        11: mouth
        12: u_lip
        13: l_lip
        14: neck
        15: neck_l
        16: cloth
        17: hair
        18: hat
        '''

        # Create binary mask for face-related classes
        face_classes = [1,2,3,4,5,6,7,8,10,11,12,13]
        binary_masks = torch.zeros_like(seg_masks)
        for class_idx in face_classes:
            binary_masks[seg_masks == class_idx] = 1
        seg_masks = binary_masks                         # [32, 1, 40, 40]
        
        #####

        # ### --- softmax 실험 ---
        # seg_softmax = F.softmax(results_list, dim=1)[:, 1:14] # 1~13 채널에 대한 softmax [2, 13, 64, 64]
        # x = torch.cat([x, seg_softmax], dim=1)              # [2, 256+13, 64, 64]

        # ### --- bbox 만들기 ---
        # B = seg_masks.shape[0]
        # bboxes = []
        
        # # 배치별로 처리
        # for b in range(B):
        #     mask = seg_masks[b].squeeze().cpu().numpy().astype(np.uint8)  # [H, W]
        #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #     # 가장 큰 contour만 bounding box로 그리자 (skin이 여러 조각일 수 있으므로)
        #     if len(contours) > 0:
        #         # 가장 큰 contour
        #         largest_contour = max(contours, key=cv2.contourArea)
        #         # bounding rectangle
        #         bx, by, bw, bh = cv2.boundingRect(largest_contour)
        #         bboxes.append([bx, by, bx+bw, by+bh])
        #         # print(f"Batch {b} Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        #     else:
        #         bboxes.append([0, 0, 0, 0])  # 빈 contour일 경우 0으로 채움

        # ### --- 중간 시각화 ---
        # seg_masks = seg_masks.float()
        # resized_masks = []
        # resized_bboxes = []  # bbox 시각화를 위한 리스트
        
        # for i in range(seg_masks.shape[0]):
        #     # # --- Train ---
        #     # size = batch_data_samples['ori_shapes'][i]
        #     # --- Test --- 
        #     size = batch_data_samples[i].ori_shape

        #     # Mask resizing
        #     mask = F.interpolate(
        #         seg_masks[i:i+1], 
        #         size=size,  # (H,W)
        #         mode="bilinear",
        #         align_corners=True
        #     )
            
        #     # Convert tensor to numpy array
        #     mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        #     resized_masks.append(mask_np)
            
        #     # bbox resizing
        #     if bboxes[i] != [0, 0, 0, 0]:
        #         orig_h, orig_w = size
        #         curr_h, curr_w = seg_masks.shape[2:4]
        #         scale_x = orig_w / curr_w
        #         scale_y = orig_h / curr_h
        #         bx, by, bx2, by2 = bboxes[i]
        #         resized_bbox = [
        #             int(bx * scale_x),
        #             int(by * scale_y),
        #             int(bx2 * scale_x),
        #             int(by2 * scale_y)
        #         ]
        #         resized_bboxes.append(resized_bbox)
        #     else:
        #         resized_bboxes.append([0, 0, 0, 0])

        # # Convert resized masks to numpy arrays and process each mask
        # processed_masks = []
        # processed_images = []
        
        # for i, mask in enumerate(resized_masks):

        #     # # --- Train ---
        #     # size = batch_data_samples['ori_shapes'][i]
        #     # --- Test --- 
        #     size = batch_data_samples[i].ori_shape
            
        #     # Convert to PIL Image
        #     mask_pil = Image.fromarray(mask)
            
        #     # Get original image size from batch_data_samples
        #     original_size = size
            
        #     # Resize mask to original image size using nearest neighbor interpolation
        #     restored_mask = mask_pil.resize(
        #         (original_size[1], original_size[0]), # PIL uses (width, height)
        #         resample=Image.NEAREST
        #     )
            
        #     # Convert back to numpy array
        #     processed_mask = np.array(restored_mask)
        #     processed_masks.append(processed_mask)

        # vis_parsing_maps(
        #     # image = batch_data_samples['imgs_path'],
        #     image = [batch.img_path for batch in batch_data_samples],
        #     bboxes = resized_bboxes,
        #     seg_masks = processed_masks,
        #     save_image = True,
        #     save_dir='mmyolo/models/bisnet_gaze/visualize_mask_bbox'
        # )

        # ### --- 시각화 끝 ---

        x = x + (x * seg_masks)                                # [B, 256, 64, 64]

        # x = self.conv(x)                                       # [B, 256, 16, 16]
        
        B, C, H ,W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = self.self_attn(q=x, k=x, v=x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # bbox_embedding = self.prompt_encoder(bboxes, batch_data_samples, training)
        # bbox_embedding = bbox_embedding.mean(dim=2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        # fused_feature = torch.cat([x, bbox_embedding], dim=1)  # [B, 256, 16, 16]

        pred_xyz = self.gaze_head_module(x)

        # return pred_xyz, pred_p3p4p5
        return pred_xyz
    
    # confidence 8 실험
    def loss(self, x: Tensor, results_list, batch_data_samples: dict) -> SampleList:
        """
            backpropagation에서 가중치 업데이트하는데 사용되는 loss 계산
        """

        gaze_labels = batch_data_samples['gazes_labels']  # gaze gt값

        # pred_xyz, pred_p3p4p5 = self(x, results_list, batch_data_samples, training = True)
        # l_xyz, l_p3p4p5, l_heatmap, l_heatmap_p3p4p5 = self.loss_gaze(pred_xyz, pred_p3p4p5, gaze_labels)

        # # neck에서 feature 하나만 사용
        # pred_xyz = self(x, results_list, batch_data_samples, training = True)
        # l_xyz, l_heatmap = self.loss_gaze(pred_xyz, gaze_labels)

        ### reg head
        pred_xyz = self(x, results_list, batch_data_samples, training = True)
        l_xyz, l_heatmap = self.loss_gaze(pred_xyz, gaze_labels)

        # return l_xyz, l_p3p4p5, l_heatmap, l_heatmap_p3p4p5
        return l_xyz, l_heatmap
    
    # confidence 8 실험
    def predict(self, x: Tensor, results_list, batch_data_samples) -> SampleList:

        # # results_list에서 최대 신뢰도 값을 가진 값만 추출
        # filtered_results_list = []
        # for list in results_list:
        #     filtered_results_list.append(self.filter_results(list))

        # pred_xyz, pred_p3p4p5 = self(x, results_list, batch_data_samples, training = False)
        # pred_xyz = self(x, results_list, batch_data_samples, training = False)
        pred_xyz = self(x, results_list, batch_data_samples, training = False)       

        final_gaze_xyz=[]
        for gaze_pred in pred_xyz:
            final_gaze_xyz.append(InstanceData(gazes = gaze_pred))

        return final_gaze_xyz
    
    def loss_by_feat(self,x):
        # 이 함수 없으면 오류남
        # 주어진 feature map 과  gt를 기반으로 손실 계산
        # 이 부분에서 loss 로 넘어감
        return x
    