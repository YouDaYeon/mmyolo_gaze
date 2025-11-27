"""
Author: Yakhyokhuja Valikhujaev
Date: 2024-08-07
Description: BiSeNet Model Implementation
Copyright (c) 2024 Yakhyokhuja Valikhujaev. All rights reserved.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# from .resnet import resnet18, resnet34
from mmyolo.models.backbones.resnet import ResNet   ######### mmyolo.models.backbones.resnet.ResNet 사용
from typing import Union, Optional, Tuple 

from mmyolo.registry import MODELS


class ConvBNReLU(nn.Module):
    """Standard Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            inplace: bool = True,
            bias: bool = False,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class BiSeNetOutput(nn.Module):
    """BiSeNet Output"""

    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, frozen: bool = False) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
        )
        self.conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=False,
        )

        # Freeze 설정
        if frozen:
            self._freeze_module()

    def _freeze_module(self):
        """Freeze all parameters of this module"""
        for param in self.parameters():
            param.requires_grad = False
        # BatchNorm 레이어들을 eval 모드로 설정
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        x = self.conv(x)
        return x

    def is_frozen(self):
        """Check if all parameters are frozen"""
        return all(not param.requires_grad for param in self.parameters())


class AttentionRefinementModule(nn.Module):
    """Attention Refinement Module """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv_block(x)
        
        feat_shape = [int(t) for t in feat.size()[2:]]
        pool = F.avg_pool2d(feat, feat_shape)
        # pool = F.avg_pool2d(feat, feat.size()[2:]) # gives error when converting to onnx due to dynamic size

        attention = self.attention(pool)
        out = torch.mul(feat, attention)
        return out

class ContextPath(nn.Module):
    """Context Path Module or Multi-Scale Feature Pyramid Module"""

    def __init__(self, backbone_name: str = "resnet18", frozen_stages: int = -1) -> None:
        super().__init__()
        if backbone_name == "resnet18":
            self.backbone = ResNet(depth=18)           ######### mmyolo.models.backbones.resnet.ResNet 사용
        elif backbone_name == "resnet34":
            self.backbone = ResNet(depth=34)
        else:
            raise Exception(f"Available backbone modules: resnet18, resnet34")

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        self.arm16 = AttentionRefinementModule(in_channels=256, out_channels=128)
        self.arm32 = AttentionRefinementModule(in_channels=512, out_channels=128)
        self.conv_head32 = ConvBNReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv_head16 = ConvBNReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv_avg = ConvBNReLU(in_channels=512, out_channels=128, kernel_size=1, stride=1)

    def _freeze_stages(self):
        """Freeze stages of the backbone"""
        if self.frozen_stages >= 0:
            # backbone의 모든 파라미터를 freeze
            for param in self.backbone.parameters():
                param.requires_grad = False
            # BatchNorm 레이어들을 eval 모드로 설정
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.track_running_stats = False
            print(f"{self.__class__.__name__} backbone has been frozen")

    def is_frozen(self):
        """Check if backbone parameters are frozen"""
        return all(not param.requires_grad for param in self.backbone.parameters())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # ResNet backbone에서 다중 스케일 특징 추출
        # feat8: 1/8 스케일 특징맵 
        # feat16: 1/16 스케일 특징맵
        # feat32: 1/32 스케일 특징맵
        _, feat8, feat16, feat32 = self.backbone(x) 

        # 각 특징맵의 공간 차원(높이, 너비) 저장
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        # 전역 컨텍스트 정보 추출을 위한 전역 평균 풀링
        feat32_shape = [int(t) for t in feat32.size()[2:]]
        avg = F.avg_pool2d(feat32, feat32_shape)              # [32, 512, 1, 1]
        
        # 전역 컨텍스트 특징을 채널 수 조정 및 업샘플링
        avg = self.conv_avg(avg)  # 채널 수를 512->128로 줄임 
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")  # 1/32 스케일로 업샘플링  [32, 128, 10, 10]

        # 1/32 스케일 특징 처리
        feat32_arm = self.arm32(feat32)  # Attention Refinement Module 적용
        feat32_sum = feat32_arm + avg_up  # 전역 컨텍스트와 결합
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode="nearest")  # 1/16 스케일로 업샘플링
        feat32_up = self.conv_head32(feat32_up)  # 최종 특징 조정

        # 1/16 스케일 특징 처리
        feat16_arm = self.arm16(feat16)  # Attention Refinement Module 적용
        feat16_sum = feat16_arm + feat32_up  # 상위 레벨 특징과 결합
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode="nearest")  # 1/8 스케일로 업샘플링
        feat16_up = self.conv_head16(feat16_up)  # 최종 특징 조정

        # 세 가지 스케일의 특징맵 반환
        # feat8: Spatial Path의 상세 특징 (1/8)
        # feat16_up: 중간 레벨 컨텍스트 특징 (1/8로 업샘플링됨)
        # feat32_up: 고레벨 컨텍스트 특징 (1/16으로 업샘플링됨)
        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module"""

    def __init__(self, in_channels: int, out_channels: int, frozen: bool = False) -> None:
        super().__init__()

        self.conv_block = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Freeze 설정
        if frozen:
            self._freeze_module()

    def _freeze_module(self):
        """Freeze all parameters of this module"""
        for param in self.parameters():
            param.requires_grad = False
        # BatchNorm 레이어들을 eval 모드로 설정
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False
        print(f"{self.__class__.__name__} has been frozen")

    def is_frozen(self):
        """Check if all parameters are frozen"""
        return all(not param.requires_grad for param in self.parameters())

    def forward(self, fsp: Tensor, fcp: Tensor) -> Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.conv_block(fcat)
        
        feat_shape = [int(t) for t in feat.size()[2:]]
        attention = F.avg_pool2d(feat, feat_shape)
        # attention = F.avg_pool2d(feat, feat.size()[2:]) # gives error when converting to onnx due to dynamic size
        
        attention = self.conv1(attention)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        feat_attention = torch.mul(feat, attention)
        feat_out = feat_attention + feat
        return feat_out

@MODELS.register_module()
class BiSeNet(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet18"):
        super().__init__()
        # Feature Pyramid Network (Context Path)를 초기화
        self.fpn = ContextPath(backbone_name=backbone_name)
        # Feature Fusion Module을 초기화 - Spatial Path와 Context Path의 특징을 결합
        self.ffm = FeatureFusionModule(in_channels=256, out_channels=256)

        # 최종 출력을 위한 컨볼루션 레이어들
        # 주요 출력 경로
        self.conv_out = BiSeNetOutput(in_channels=256, mid_channels=256, num_classes=num_classes)
        # 보조 출력 경로 (1/16 스케일)
        self.conv_out16 = BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)
        # 보조 출력 경로 (1/32 스케일) 
        self.conv_out32 = BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)

    def forward(self, x):
        # 입력 이미지의 높이와 너비 저장
        h, w = x.size()[2:]
        
        # Context Path를 통해 다중 스케일 특징 추출
        # feat_res8: Spatial Path 특징 (1/8 스케일)
        # feat_cp8: Context Path 특징 (1/8 스케일)
        # feat_cp16: Context Path 특징 (1/16 스케일)
        feat_res8, feat_cp8, feat_cp16 = self.fpn(x)
        
        # Spatial Path와 Context Path 특징을 Feature Fusion Module에서 결합
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        # 각 스케일에서 세그멘테이션 예측 수행
        feat_out_c = self.conv_out(feat_fuse)      # 주요 예측
        feat_out16 = self.conv_out16(feat_cp8)   # 보조 예측 (1/16)
        feat_out32 = self.conv_out32(feat_cp16)  # 보조 예측 (1/32)

        # 모든 예측을 원본 이미지 크기로 업샘플링
        feat_out = F.interpolate(feat_out_c, (h, w), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (h, w), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (h, w), mode="bilinear", align_corners=True)

        # 주요 예측과 보조 예측들을 반환
        return feat_fuse, feat_out_c, [feat_out, feat_out16, feat_out32]
