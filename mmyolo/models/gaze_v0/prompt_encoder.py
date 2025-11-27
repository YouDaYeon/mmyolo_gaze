# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from collections import defaultdict

import torch.nn.functional as F

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,                       # bbox를 변환할 feature map 차원
        # input_image_size: Tuple[int, int]     # 원본 이미지 크기 (H, W)
    ) -> None:
        """
        BBox를 입력받아, 이를 Feature Map(sparse_embeddings)으로 변환하는 역할을 수행
        PositionEmbeddingRandom을 이용해 BBox 좌표를 학습 가능한 벡터로 변환
        Corner Point마다 다른 Embedding을 적용하여 BBox의 표현력을 증가시킴

        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  # 2 box corners
        # self.num_point_embeddings: int = 4  # 4 bbox
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]  # 각 Coner point에 대해 학습 가능한 Embedding Vector를 할당하여 차별적인 표현을 학습 가능하도록 함
        self.point_embeddings = nn.ModuleList(point_embeddings)

        # face_point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]  
        # self.face_point_embeddings = nn.ModuleList(face_point_embeddings)
        # eye_point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        # self.eye_point_embeddings = nn.ModuleList(eye_point_embeddings)
        
        # ##### 줄무늬 패턴 개선 실험
        # self.attention_embedding = nn.Parameter(torch.randn(self.num_point_embeddings, 100))

        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _embed_boxes(self, boxes: torch.Tensor, input_image_size: torch.Tensor) -> torch.Tensor:
        """
        bbox 좌표를 feature map으로 변환하는 함수
        Embeds box prompts.

        Arguments:
            boxes (torch.Tensor): BBox 좌표 (batch_size, 4) (x_min, y_min, x_max, y_max) 
            input_image_size (torch.Tensor): 각 배치마다 다른 원본 이미지 크기 (batch_size, 2)

        Returns:
            torch.Tensor: 변환된 BBox 좌표 (batch_size, 2, embed_dim)
        """
        boxes = boxes + 0.5  # pixel 정중앙으로 정렬
        coords = boxes.reshape(-1, 2, 2)  # (bsz x 2 x 2) 코너 좌표 추출
        corner_embedding = self.pe_layer.forward_with_coords(coords, input_image_size) # 좌표를 Position Ecoding으로 위치 정보를 포함한 벡터로 변환 (bs*3, 2, embed_dim)
        corner_embedding[:, 0, :] += self.point_embeddings[0].weight  # x_min, y_min 좌표에 대해 Embedding 추가 (corner 정보를 더 잘 학습할 수 있도록)
        corner_embedding[:, 1, :] += self.point_embeddings[1].weight  # x_max, y_max 좌표에 대해 Embedding 추가
        # corner_embedding[:, 2, :] += self.point_embeddings[2].weight
        # corner_embedding[:, 3, :] += self.point_embeddings[3].weight  

        # # bbox 없을 때 corner_embedding=0 유지
        # batch_size = boxes.shape[0]
        # corner_embedding = torch.zeros((batch_size, 2, self.embed_dim), device=self.device)
        
        # # 각 배치별로 처리
        # for b in range(batch_size):
        #     # 현재 배치의 bbox가 모두 0인지 확인
        #     if torch.all(boxes[b] == 0):
        #         continue  # 0으로 초기화된 corner_embedding 유지
            
        #     # 유효한 bbox인 경우 처리
        #     box = boxes[b] + 0.5  # pixel 정중앙으로 정렬
        #     coords = box.reshape(1, 2, 2)  # (1, 2, 2) 코너 좌표 추출
        #     current_embedding = self.pe_layer.forward_with_coords(
        #         coords, 
        #         input_image_size[b:b+1]
        #     )  # (1, 2, embed_dim)
            
        #     # point embeddings 추가
        #     current_embedding[:, 0, :] += self.point_embeddings[0].weight  # x_min, y_min 좌표에 대해 Embedding 추가
        #     current_embedding[:, 1, :] += self.point_embeddings[1].weight  # x_max, y_max 좌표에 대해 Embedding 추가
            
        #     corner_embedding[b] = current_embedding[0]
        
        return corner_embedding  # (batch_size, 2, embed_dim)

    # def _embed_boxes(self, boxes: torch.Tensor, input_image_size: torch.Tensor, mode: str) -> torch.Tensor:
    #     """
    #     bbox 좌표를 feature map으로 변환하는 함수
    #     Embeds box prompts.

    #     Arguments:
    #         boxes (torch.Tensor): BBox 좌표 (batch_size, 4) (x_min, y_min, x_max, y_max) 
    #         input_image_size (torch.Tensor): 각 배치마다 다른 원본 이미지 크기 (batch_size, 2)

    #     Returns:
    #         torch.Tensor: 변환된 BBox 좌표 (batch_size, 2, embed_dim)
    #     """

    #     boxes = boxes + 0.5  # pixel 정중앙으로 정렬
    #     coords = boxes.reshape(-1, 2, 2)  # (bsz x 2 x 2) 코너 좌표 추출
    #     corner_embedding = self.pe_layer.forward_with_coords(coords, input_image_size) # 좌표를 Position Ecoding으로 위치 정보를 포함한 벡터로 변환 (bs*3, 2, embed_dim)
        
    #     if mode == "face":
    #         corner_embedding[:, 0, :] += self.face_point_embeddings[0].weight 
    #         corner_embedding[:, 1, :] += self.face_point_embeddings[1].weight 
    #     elif mode == "eye":
    #         corner_embedding[:, 0, :] += self.eye_point_embeddings[0].weight 
    #         corner_embedding[:, 1, :] += self.eye_point_embeddings[1].weight 
        
    #     return corner_embedding # (bs*3, 2, embed_dim) (32, 2, 576)

    def _get_batch_size(
        self,
        boxes: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if boxes is not None:
            return boxes.shape[0]
        else:
            return 1

    # def get_dense_pe(self) -> torch.Tensor:
    #     """
    #     Sam_decoder 에서 image_pe로 사용. batch_size를 고려하지 않고 고정된 (C, H, W) 포지셔널 인코딩을 생성.
    #     Returns the positional encoding used to encode point prompts,
    #     applied to a dense set of points the shape of the image encoding.

    #     Returns:
    #       torch.Tensor: Positional encoding with shape
    #         1x(embed_dim)x(embedding_h)x(embedding_w)
    #     """
    #     return self.pe_layer(size=(10, 10)).unsqueeze(0)
    #     # return self.pe_layer(size=(40, 40)).unsqueeze(0)  # p3p4p5

    def get_dense_pe_2(self, i) -> torch.Tensor:

        if i == 0:
            return self.pe_layer(size=(40, 40)).unsqueeze(0)
        elif i == 1:
            return self.pe_layer(size=(20, 20)).unsqueeze(0)
            # return self.pe_layer(size=(40, 40)).unsqueeze(0)     # bisnet(resnet) 실험
        elif i == 2:
            return self.pe_layer(size=(10, 10)).unsqueeze(0)
            # return self.pe_layer(size=(20, 20)).unsqueeze(0)     # bisnet(resnet) 실험
    # def get_dense_pe_2(self, i) -> torch.Tensor:

    #     if i == 0:
    #         return self.pe_layer(size=(80, 80)).unsqueeze(0)
    #     elif i == 1:
    #         return self.pe_layer(size=(40, 40)).unsqueeze(0)
    #     elif i == 2:
    #         return self.pe_layer(size=(20, 20)).unsqueeze(0)
    
    def forward(
        self,
        x,
        results_list,
        batch_data_samples,
        training: bool,
        # boxes: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        bbox를 입력받아 _embed_boxes()를 호출하여 feature map으로 변환
        결과적으로 sparse_embeddings라는 최종 feature map을 반환
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = len(results_list)

        # # label이 input으로 들어오는 경우 실험 : results_list가 batch_data_samples에 bbox labels임.
        # if training:
        #     bs = len(results_list) // 3
        # else:
        #     bs = len(results_list)

        # face_bboxes = torch.zeros((bs, 4), dtype=torch.float32, device=self.device)  # 기본값 (0, 0, 0, 0)
        # eye_bboxes = torch.zeros((bs, 2, 4), dtype=torch.float32, device=self.device)
        # eye_count = defaultdict(int)                         # batch_idx마다 개수 추적

        if training:
            input_image_size = torch.tensor(
                [list(meta['batch_input_shape']) for meta in batch_data_samples['img_metas']], dtype=torch.int, device=self.device)

            # # label이 input으로 들어오는 경우 실험
            # for result in results_list:
            #     batch_idx = int(result[0].item())
            #     label = int(result[1].item())
            #     bbox = result[2:]
                
            #     if label == 0:
            #         face_bboxes[batch_idx] = bbox
            #     elif label == 1:
            #         count = eye_count[batch_idx]
            #         eye_bboxes[batch_idx, count] = bbox
            #         eye_count[batch_idx] += 1

        else:
            input_image_size = torch.tensor(
                [list(batch_data.ori_shape) for batch_data in batch_data_samples], dtype=torch.int, device=self.device)

            # # label이 input으로 들어오는 경우 실험
            # for batch_idx, batch_data in enumerate(batch_data_samples):
            #     labels = batch_data.gt_instances.labels.tolist()
            #     bboxes = batch_data.gt_instances.bboxes
            #     for label, bbox in zip(labels, bboxes):
            #         if label == 0:
            #             face_bboxes[batch_idx] = bbox
            #         elif label == 1:
            #             count = eye_count[batch_idx]
            #             eye_bboxes[batch_idx, count] = bbox  
            #             eye_count[batch_idx] += 1
        
        face_bboxes = torch.zeros((bs, 4), dtype=torch.float32, device=self.device)  # 기본값 (0, 0, 0, 0)
        eye_bboxes = torch.zeros((bs, 2, 4), dtype=torch.float32, device=self.device)

        for batch_idx, result in enumerate(results_list):
            eye_count = 0

            for label, bbox in zip(result.labels, result.bboxes):
                if label == 0 : 
                    face_bboxes[batch_idx] = bbox
                elif label == 1:
                    eye_bboxes[batch_idx, eye_count] = bbox
                    eye_count += 1                 # 눈이 두개면 두번째 eye bbox를 저장하는 위치

        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.device)

        # # _embed_boxes() 호출
        face_bboxes_embeddings = self._embed_boxes(face_bboxes, input_image_size)  # (batch_size * num_boxes, 2, embed_dim)
        # first_eye_bboxes_embeddings = self._embed_boxes(eye_bboxes[:,0,:], input_image_size)  # (batch_size * num_boxes, 2, embed_dim)
        # second_eye_bboxes_embeddings = self._embed_boxes(eye_bboxes[:,1,:], input_image_size)  # (batch_size * num_boxes, 2, embed_dim)
        
        # # _embed_boxes() 호출 : face와 eye 각각
        # face_bboxes_embeddings = self._embed_boxes(face_bboxes, input_image_size, mode="face")  # (batch_size * num_boxes, 2, embed_dim)
        # first_eye_bboxes_embeddings = self._embed_boxes(eye_bboxes[:,0,:], input_image_size, mode="eye")  # (batch_size * num_boxes, 2, embed_dim)
        # second_eye_bboxes_embeddings = self._embed_boxes(eye_bboxes[:,1,:], input_image_size, mode="eye")  # (batch_size * num_boxes, 2, embed_dim)

        # 4(batch_size, max_bboxes * 2, embed_dim) 형태로 변환
        face_bboxes_embeddings = face_bboxes_embeddings.view(bs, -1, self.embed_dim)  # (bs, num_boxes * 2, embed_dim)
        # first_eye_bboxes_embeddings = first_eye_bboxes_embeddings.view(bs, -1, self.embed_dim)  # (bs, num_boxes * 2, embed_dim)
        # second_eye_bboxes_embeddings = second_eye_bboxes_embeddings.view(bs, -1, self.embed_dim)  # (bs, num_boxes * 2, embed_dim)


        # total_bboxes_embeddings = face_bboxes_embeddings + first_eye_bboxes_embeddings + second_eye_bboxes_embeddings
        # total_bboxes_embeddings = face_bboxes_embeddings
        # 최종 sparse_embeddings에 추가
        # sparse_embeddings = torch.cat([sparse_embeddings, total_bboxes_embeddings], dim=1)
        # sparse_embeddings = torch.cat([sparse_embeddings, face_bboxes_embeddings, first_eye_bboxes_embeddings, second_eye_bboxes_embeddings], dim=1)
        
        # sparse_embeddings = sparse_embeddings.permute(0, 2, 1)   # 추가

        # # 정규화 후 시각화
        # x_norm = self.min_max_normalize(x)
        # sparse_embeddings_norm = 5 * self.min_max_normalize(sparse_embeddings)

        # return x_norm + sparse_embeddings_norm
        # return sparse_embeddings
        # return face_bboxes_embeddings.permute(0, 2, 1), first_eye_bboxes_embeddings.permute(0, 2, 1), second_eye_bboxes_embeddings.permute(0, 2, 1)
        return face_bboxes_embeddings.permute(0, 2, 1)

    # def forward(
    #     self,
    #     x,
    #     batch_data_samples,
    #     training: bool,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     gaze5(resnet) : bbox_labels 사용
    #     """
    #     if training:
    #         input_image_size = torch.tensor(
    #             [list(meta['batch_input_shape']) for meta in batch_data_samples['img_metas']], dtype=torch.int, device=self.device)
    #         bbox_labels = batch_data_samples['bboxes_labels']
    #     else:
    #         input_image_size = torch.tensor(
    #             [list(batch_data.ori_shape) for batch_data in batch_data_samples], dtype=torch.int, device=self.device)
    #         bbox_labels = []
    #         for i, batch_data in enumerate(batch_data_samples):
    #             labels = batch_data.gt_instances.labels.tolist()
    #             bboxes = batch_data.gt_instances.bboxes.tolist()
    #             for label, bbox in zip(labels, bboxes):
    #                 bbox_labels.append([i, label] + bbox)
    #         bbox_labels = torch.tensor(bbox_labels, device=self.device)
            
    #     face_bboxes = bbox_labels[bbox_labels[:, 1] == 0][:,2:]
    #     eye_bboxes = bbox_labels[bbox_labels[:, 1] == 1][:,2:]

    #     bs = len(face_bboxes)

    #     face_bboxes_embeddings = self._embed_boxes(face_bboxes, input_image_size)      # (bs, 2, embed_dim)

    #     return face_bboxes_embeddings.permute(0, 2, 1)

    # def forward(
    #     self,
    #     bboxes,
    #     batch_data_samples,
    #     training: bool,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     gaze_seg : seg map에서 나온 bbox 좌표 사용
    #     """
    #     bs = len(bboxes)

    #     if training:
    #         input_image_size = torch.tensor(
    #             [list(meta['batch_input_shape']) for meta in batch_data_samples['img_metas']], dtype=torch.int, device=self.device)
    #     else:
    #         input_image_size = torch.tensor(
    #             [list(batch_data.ori_shape) for batch_data in batch_data_samples], dtype=torch.int, device=self.device)
        
    #     face_bboxes = torch.tensor(bboxes, device=self.device)

    #     face_bboxes_embeddings = self._embed_boxes(face_bboxes, input_image_size)      # (bs, 2, embed_dim)

    #     return face_bboxes_embeddings.permute(0, 2, 1)
    
    # Min-Max 정규화 (각 텐서를 0~1 범위로 정규화)
    def min_max_normalize(self,tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # # 리스트를 텐서로 변환
    # def to_tensor(self, bbox_list, batch_size=32):
    #     # 초기 텐서
    #     bbox_tensor = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)
    #     for batch_idx, bbox in enumerate(bbox_list):
    #         if batch_idx < batch_size:
    #             bbox_tensor[batch_idx] = torch.tensor(bbox, dtype=torch.float32, device=self.device)
    #     return bbox_tensor

class PositionEmbeddingRandom(nn.Module):
    """
    bbox의 corner point 좌표를 position encoding을 사용해 feature map으로 변환
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),    # 2는 x, y 좌표를 의미
        )

    # def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
    #     """Positionally encode points that are normalized to [0,1].
    #        Position Encoding으로 변환.
    #        coords: x1 y1
    #                x2 y2"""
    #     # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
    #     coords = 2 * coords - 1                                                 # 정규화 [-1,1] 범위로 변환 (32, 2, 2)

    #     # coords 좌표 reshape
    #     # 각 좌표를 분리 (x_min, y_min, x_max, y_max)
    #     x_min, y_min = coords[:, 0, 0], coords[:, 0, 1]
    #     x_max, y_max = coords[:, 1, 0], coords[:, 1, 1]
    #     # # 각 코너 좌표 생성
    #     # coord1 = torch.stack([x_min, y_min], dim=1)  # (32, 2)
    #     # coord2 = torch.stack([x_max, y_min], dim=1)  # (32, 2)
    #     # coord3 = torch.stack([x_min, y_max], dim=1)  # (32, 2)
    #     # coord4 = torch.stack([x_max, y_max], dim=1)  # (32, 2)
    #     # # 4개의 좌표를 하나의 텐서로 합침
    #     # coords = torch.stack([coord1, coord2, coord3, coord4], dim=1)  # (32, 4, 2)

    #     # # center 좌표 생성
    #     center_x = (x_max-x_min) / 2 + x_min
    #     center_y = (y_max-y_min) / 2 + y_min
    #     coord_center = torch.stack([center_x, center_y], dim=1)  # (32, 2)
    #     coord1 = torch.stack([x_min, y_min], dim=1)  # (32, 2)
    #     coord2 = torch.stack([x_max, y_max], dim=1)  # (32, 2)
    #     coords = torch.stack([coord1, coord2, coord_center], dim=1)  # (32, 3, 2) 

    #     coords = coords @ self.positional_encoding_gaussian_matrix              # 랜덤 가우시안 행렬 적용
    #     coords = 2 * np.pi * coords                                             # 주기성 부여
    #     # outputs d_1 x ... x d_n x C shape
    #     return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)        # Sin & Cos 변환
    
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1].
           Position Encoding으로 변환.
           coords: x1 y1
                   x2 y2"""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1                                                 # 정규화 [-1,1] 범위로 변환 (32, 2, 2)
        coords = coords @ self.positional_encoding_gaussian_matrix              # 랜덤 가우시안 행렬 적용
        coords = 2 * np.pi * coords                                             # 주기성 부여
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)        # Sin & Cos 변환
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5  # (H, W)
        x_embed = grid.cumsum(dim=1) - 0.5  # (H, W)
        y_embed = y_embed / h               # [0, 1] 범위로 정규화
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # (H, W, 2)
        # pe = self._pe_encoding_orig(torch.stack([x_embed, y_embed], dim=-1))  # (H, W, 2)
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: torch.Tensor
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1].
           Position Encoding 적용 전 [0,1]로 정규화 진행."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[:, 1].unsqueeze(-1)  # x 좌표 정규화
        coords[:, :, 1] = coords[:, :, 1] / image_size[:, 0].unsqueeze(-1)  # y 좌표 정규화
        return self._pe_encoding(coords.to(torch.float))   # B x N x C