import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from typing import List, Optional
from mmdet.structures.bbox import get_box_tensor
from mmcv.ops import batched_nms
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmyolo.registry import MODELS

import math

# nn.Module을 상속하여 만듦. __init__에서 cfg 받아와 초기화.
@MODELS.register_module()
class FilteringLayer(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(FilteringLayer, self).__init__()
        self.cfg = cfg

    # 기존 dy_filtering 함수
    def forward(self,
                results: InstanceData,
                rescale: bool = False,
                with_nms: bool = True,
                img_meta: Optional[dict] = None) -> InstanceData:
        
        new_results = InstanceData()
        

        if with_nms and results.bboxes.numel() > 0: # numel () :텐서에 있는 총 요소의 개수 
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, self.cfg.nms)
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[:self.cfg.max_per_img]

        labels = results.labels
        scores = results.scores
        bboxes = results.bboxes

        face_bboxes = []
        eye_bboxes = []
        face_scores = []
        eye_scores = []
        for label, bbox, score in zip(labels, bboxes, scores):

            if label == 0:
                face_bboxes.append(bbox)
                face_scores.append(score)
            elif label == 1:
                eye_bboxes.append(bbox)
                eye_scores.append(score)

        face_bboxes = torch.stack(face_bboxes) if face_bboxes else torch.empty((0, 4), device=bboxes.device)
        eye_bboxes = torch.stack(eye_bboxes) if eye_bboxes else torch.empty((0, 4), device=bboxes.device)

        face_bboxes_with_conf = [(bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item(), score.item()) 
                                 for bbox, score in zip(face_bboxes, face_scores)]
        eye_bboxes_with_conf = [(bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item(), score.item()) 
                                for bbox, score in zip(eye_bboxes, eye_scores)]

        selected_face_bbox, selected_eye_bboxes = self.filter_bboxes(face_bboxes_with_conf, eye_bboxes_with_conf)
        # selected_face_bbox, selected_eye_bboxes = self.filter_bboxes_4(img_meta, face_bboxes_with_conf, eye_bboxes_with_conf)

        # if selected_face_bbox is not None:
        #     new_labels = [0]  # Label 0 for face
        #     new_scores = [selected_face_bbox[4]]
        #     new_bboxes = [selected_face_bbox[:4]]

        #     for eye_bbox in selected_eye_bboxes:
        #         new_labels.append(1)  # Label 1 for eye
        #         new_scores.append(eye_bbox[4])
        #         new_bboxes.append(eye_bbox[:4])
                
        #     new_results.labels = torch.tensor(new_labels, device=bboxes.device)
        #     new_results.scores = torch.tensor(new_scores, device=bboxes.device)
        #     new_results.bboxes = torch.tensor(new_bboxes, device=bboxes.device)
        # else:
        #     new_results.labels = torch.empty((0,), device=bboxes.device)
        #     new_results.scores = torch.empty((0,), device=bboxes.device)
        #     new_results.bboxes = torch.empty((0, 4), device=bboxes.device)

        if selected_face_bbox is not None or len(selected_eye_bboxes) > 0:
            new_labels, new_scores, new_bboxes = [], [], []

            if selected_face_bbox is not None:
                new_labels.append(0)  # Label 0 for face
                new_scores.append(selected_face_bbox[4])
                new_bboxes.append(selected_face_bbox[:4])

            if selected_eye_bboxes is not None:
                for eye_bbox in selected_eye_bboxes:
                    new_labels.append(1)  # Label 1 for eye
                    new_scores.append(eye_bbox[4])
                    new_bboxes.append(eye_bbox[:4])

            # lists -> tensors
            new_results.labels = torch.tensor(new_labels, device=bboxes.device)
            new_results.scores = torch.tensor(new_scores, device=bboxes.device)
            new_results.bboxes = torch.tensor(new_bboxes, device=bboxes.device)

        else:
            new_results.labels = torch.empty((0,), device=bboxes.device)
            new_results.scores = torch.empty((0,), device=bboxes.device)
            new_results.bboxes = torch.empty((0, 4), device=bboxes.device)

        return new_results
        # return results

    def filter_bboxes(self, face_bboxes, eye_bboxes):
        selected_face_bbox = None
        selected_eye_bboxes = []

        # 조건 2
        if len(face_bboxes) < 1 and len(eye_bboxes) < 1:
            return selected_face_bbox, selected_eye_bboxes
        
        # 얼굴 없을 때 여기를 실험중. 눈도 빈박스로 리턴
        if len(face_bboxes) < 1:
            selected_eye_bboxes = sorted(eye_bboxes, key=lambda x: x[4], reverse=True)[:2]
            return selected_face_bbox, selected_eye_bboxes

        if len(eye_bboxes) < 1:
            selected_face_bbox = sorted(face_bboxes, key=lambda x: x[4], reverse=True)[0]
            return selected_face_bbox, selected_eye_bboxes

        # 얼굴 안에 눈이 존재하는지 먼저 보고, 존재하면 그 때의 얼굴 score중 가장 높은 얼굴과 눈 / 존재하지 않으면 score중 가장 높은 얼굴
        valid_face_bboxes = []
        for face_bbox in face_bboxes:
            eye_count = 0
            for eye_bbox in eye_bboxes:
                if self.is_inside(eye_bbox, face_bbox):
                    eye_count += 1
            if eye_count >= 1:
                valid_face_bboxes.append(face_bbox)

        # 얼굴 안에 눈이 존재하지 않을 때
        if len(valid_face_bboxes) == 0:
            selected_face_bbox = max(face_bboxes, key=lambda x: x[4])
            return selected_face_bbox, selected_eye_bboxes

        # 얼굴 안에 눈이 존재할 때
        selected_face_bbox = max(valid_face_bboxes, key=lambda x: x[4])

        for eye_bbox in eye_bboxes:
            if self.is_inside(eye_bbox, selected_face_bbox):
                selected_eye_bboxes.append(eye_bbox)

        # ## 눈이 겹치는 경우가 없게 처리 !!!
        # selected_eye_bboxes = sorted(selected_eye_bboxes, key=lambda x: x[4], reverse=True)
        # selected_eye_bboxes_not_overlap = []
        # selected_eye_bboxes_not_overlap.append(selected_eye_bboxes[0])
        # for i in range(1, len(selected_eye_bboxes)):
        #     # 현재 bbox가 기존의 모든 bbox들과 겹치지 않는지 확인
        #     should_add = True
        #     for existing_bbox in selected_eye_bboxes_not_overlap:
        #         if not self.is_not_overlap(selected_eye_bboxes[i], existing_bbox):
        #             should_add = False
        #             break
        #     if should_add:
        #         selected_eye_bboxes_not_overlap.append(selected_eye_bboxes[i])
        # selected_eye_bboxes = selected_eye_bboxes_not_overlap

        selected_eye_bboxes = sorted(selected_eye_bboxes, key=lambda x: x[4], reverse=True)[:2]

        return selected_face_bbox, selected_eye_bboxes

    def is_inside(self, eye_bbox, face_bbox):
        ex1, ey1, ex2, ey2 = eye_bbox[:4]
        fx1, fy1, fx2, fy2 = face_bbox[:4]

        return (fx1 <= ex1 <= fx2 and fy1 <= ey1 <= fy2 and
                fx1 <= ex2 <= fx2 and fy1 <= ey2 <= fy2)
    
    def is_not_overlap(self, eye_bbox1, eye_bbox2):
        ex1, ey1, ex2, ey2 = eye_bbox1[:4]
        ex3, ey3, ex4, ey4 = eye_bbox2[:4]
        
        # 두 박스가 겹치지 않는 경우: 
        # 1. bbox1이 bbox2의 왼쪽에 있거나
        # 2. bbox1이 bbox2의 오른쪽에 있거나  
        # 3. bbox1이 bbox2의 위쪽에 있거나
        # 4. bbox1이 bbox2의 아래쪽에 있을 때
        return (ex2 < ex3) or (ex1 > ex4) or(ey2 < ey3) or (ey1 > ey4)

    def distance_from_center(self, image_center, face_bbox):
        face_bbox_center = (face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2
        distance = math.sqrt((image_center[0] - face_bbox_center[0])**2 + (image_center[1] - face_bbox_center[1])**2)

        return distance

