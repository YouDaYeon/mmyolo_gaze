import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import math

# from .MultiHeadAttention import MultiHeadAttention

class PositionMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, results_list, batch_data_samples, training: bool):

        device = x.device
        face_position_map = self.ROI_position_extract(x, results_list, batch_data_samples, training)
        face_position_map = face_position_map.to(device)

        return face_position_map*x
        # return face_position_map*x + x
    
    def ROI_position_extract(self, x, results_list, batch_data_samples, training: bool):

        b, c, h, w = x.shape

        # p3, p4, p5 크기로 0을 채운 텐서 생성
        face_position_map = torch.zeros(b, c, h, w)
        # face_position_map = torch.zeros(b, 1, h, w)

        for batch_idx in range(b):

            # 이미지 크기
            if training:
                img_h, img_w = batch_data_samples['img_metas'][batch_idx]['batch_input_shape']
            else:
                img_h, img_w = batch_data_samples[batch_idx].ori_shape[:2]

            scale_w = w / img_w
            scale_h = h / img_h
                
            result = results_list[batch_idx]

            for label, bbox in zip(result.labels, result.bboxes):
                x1, y1, x2, y2 = bbox.clone()
                # x1 *= scale_w
                # x2 *= scale_w
                # y1 *= scale_h
                # y2 *= scale_h
                x1 = torch.floor(x1 * scale_w)   # 무조건 내림
                x2 = torch.ceil(x2 * scale_w)    # 무조건 올림
                y1 = torch.floor(y1 * scale_h)
                y2 = torch.ceil(y2 * scale_h)

                # if label == 0:  # face
                #     if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                #         face_position_map[batch_idx] = 1  # Fill entire feature map with 1s
                #     # else:
                #     #     face_position_map[batch_idx, :, int(y1):int(y2), int(x1):int(x2)] = 1
                
                if label == 1:  # eye
                    if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
                        face_position_map[batch_idx, :, int(y1):int(y2), int(x1):int(x2)] = 1
                        # face_position_map[batch_idx, :, int(y1):int(y2), int(x1):int(x2)] = 2

        return face_position_map

    # def position_map_atten(self, x, face_position_map):
        
    #     B, C, H, W = x.shape
        
    #     # # Project Q/K/V
    #     # Q = self.q_conv(face_position_map).view(B, -1, H*W).transpose(1, 2)        # B x HW x E
    #     # K = self.k_conv(x).view(B, -1, H*W).transpose(1, 2)        # B x HW x E
    #     # V = self.v_conv(x).view(B, -1, H*W).transpose(1, 2)        # B x HW x E

    #     # # Project Q/K/V
    #     # Q = self.q_conv(x).view(B, -1, H*W).transpose(1, 2)        # B x HW x E
    #     # K = self.k_conv(x).view(B, -1, H*W).transpose(1, 2)
    #     # V = self.v_conv(x).view(B, -1, H*W).transpose(1, 2)


    #     # 실험 a3
    #     Q = face_position_map.view(B, -1, H*W).transpose(1, 2)        # B x HW x E
    #     Q = self.q_conv(Q)
    #     K = x.view(B, -1, H*W).transpose(1, 2)
    #     K = self.k_conv(K)
    #     V = x.view(B, -1, H*W).transpose(1, 2)
    #     V = self.v_conv(V)




    #     # Transpose K for dot product
    #     K_t = K.transpose(1, 2)                    # B x E x HW
        
    #     # Compute attention map (HW x HW)
    #     attn = torch.bmm(Q, K_t)   # B x HW x HW
    #     attn = F.softmax(attn * self.scale, dim=-1)
        
    #     # Apply attention to V
    #     out = torch.bmm(attn, V).transpose(1, 2)   # B x E x HW
    #     out = out.view(B, -1, H, W)                # B x E x H x W
        
    #     # Project back to original channel
    #     out = self.out_conv(out)
        
    #     # Residual connection
    #     out = out + x
        
    #     return out





    # From https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    # 각 픽셀의 (x, y) 좌표를 다중 주파수의 sin/cos로 표현한 절대 2D 위치 임베딩
    # flatten으로 순서가 사라지는 attention에 공간 좌표 정보를 주입
    def positionalencoding2d(self,d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe