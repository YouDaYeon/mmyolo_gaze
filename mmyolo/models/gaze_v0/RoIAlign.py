import torch
import torch.nn as nn
from torchvision.ops import roi_align
import torch.nn.functional as F


class ROIAlignModule(nn.Module):
    def __init__(self, output_size=(7, 7), sampling_ratio=-1, aligned=True):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        
    def forward(self, features, results_list, batch_data_samples, training: bool):
        """
        Args:
            features: Tensor(B, C, H, W)
            results_list: List of detection results, each with `labels` and `bboxes`
            batch_data_samples: List of image meta info
            training: bool flag
        Returns:
            face_feats: Tensor(B, C, H, W)
            eye_feats: Tensor(B, 2, C, H, W)  ← left, right eye
        """
        b, c, h, w = features.shape
        device = features.device
        # Initialize with zeros for each batch
        face_rois = [torch.tensor([[i, 0, 0, 0, 0]], dtype=torch.float32, device=device) for i in range(b)]
        # eye_rois = []
        # for i in range(b):
        #     eye_rois.extend([
        #         torch.tensor([[i, 0, 0, 0, 0]], device=device),
        #         torch.tensor([[i, 0, 0, 0, 0]], device=device)
        #     ])
        # eye_valid_mask = []

        for batch_idx in range(b):
            # 이미지 크기
            if training:
                img_h, img_w = batch_data_samples['img_metas'][batch_idx]['batch_input_shape']
            else:
                img_h, img_w = batch_data_samples[batch_idx].ori_shape[:2]

            scale_w = w / img_w
            scale_h = h / img_h

            result = results_list[batch_idx]

            # eyes = []
            for label, bbox in zip(result.labels, result.bboxes):
                x1, y1, x2, y2 = bbox.clone()
                x1 *= scale_w
                x2 *= scale_w
                y1 *= scale_h
                y2 *= scale_h

                if label == 0:  # face
                    if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                        face_rois[batch_idx] = torch.tensor([[batch_idx, 0, 0, w, h]], device=device)
                    else:
                        face_rois[batch_idx] = torch.tensor([[batch_idx, x1, y1, x2, y2]], device=device)
                # elif label == 1:  # eye
                #     eyes.append(torch.tensor([[batch_idx, x1, y1, x2, y2]], device=device))

            # if len(eyes) == 2:
            #     eye_rois[batch_idx*2] = eyes[0]
            #     eye_rois[batch_idx*2+1] = eyes[1]
            #     eye_valid_mask.append(torch.tensor([1, 1], device=device))
            # elif len(eyes) == 1:
            #     eye_rois[batch_idx*2] = eyes[0]
            #     eye_valid_mask.append(torch.tensor([1, 0], device=device))
            # else:
            #     eye_valid_mask.append(torch.tensor([0, 0], device=device))

        # face feature 추출
        if face_rois:
            face_rois_tensor = torch.cat(face_rois, dim=0)
            face_feats = roi_align(
                features, face_rois_tensor, self.output_size,
                spatial_scale=1.0,
                sampling_ratio=self.sampling_ratio,
                aligned=self.aligned
            )

        # # eye feature 추출
        # if eye_rois:
        #     eye_rois_tensor = torch.cat(eye_rois, dim=0)  # (2B, 5)
        #     eye_feats = roi_align(
        #         features, eye_rois_tensor, self.output_size,
        #         spatial_scale=1.0,
        #         sampling_ratio=self.sampling_ratio,
        #         aligned=self.aligned
        #     )  # (2B, C, H, W)
        #     eye_feats = eye_feats.view(b, 2, c, *self.output_size)  # (B, 2, C, H, W)
        # else:
        #     eye_feats = torch.zeros((b, 2, c, *self.output_size), device=device)

        # # eye feature 평균 계산
        # # eye_valid_mask: (B, 2) with 1 or 0
        # # 예: [[1, 1], [1, 0], [0, 0]] 등
        # eye_valid_mask_tensor = torch.stack(eye_valid_mask, dim=0).to(features.device)  # (B, 2)

        # # 마스킹 및 평균 계산
        # # eye_feats: (B, 2, C, H, W)
        # mask = eye_valid_mask_tensor.view(b, 2, 1, 1, 1)  # broadcast to (B, 2, 1, 1, 1)
        # masked_eye_feats = eye_feats * mask  # invalid ROI는 0 유지

        # # 눈 개수로 나누기 (0 방지 위해 clamp)
        # valid_counts = mask.sum(dim=1).clamp(min=1)  # (B, 1, 1, 1)

        # # 평균
        # eye_feats_avg = masked_eye_feats.sum(dim=1) / valid_counts  # (B, C, H, W)

        # embedding feautre 추가
        


        # return face_feats, eye_feats_avg



        return face_feats
