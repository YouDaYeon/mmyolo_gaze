# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmyolo.models.detectors import YOLODetector
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmyolo.registry import MODELS
from mmengine.structures import InstanceData

from mmyolo.models.gaze_v0.box_filter_layer import FilteringLayer

@MODELS.register_module()
class GazeDetector_v0(YOLODetector):
    """
    기존 YOLODetector에 gaze_head 추가
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 gaze_head: ConfigType,  # 추가
                #  front_back_classifier_head: ConfigType = None,  # 추가
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        gaze_head.update(train_cfg=train_cfg)
        gaze_head.update(test_cfg=test_cfg)
        self.gaze_head = MODELS.build(gaze_head)
        # self.front_back_classifier_head = MODELS.build(front_back_classifier_head)  # 추가

        self.box_filter_layer = FilteringLayer(cfg=test_cfg)

    # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')
    
        # self.current_epoch = 0
        # self.total_epochs = 100
        # self.T1 = 10
        # self.T2 = 20
        # self.lam = 1e-3


    # def on_train_epoch_start(self, epoch):
    #     """각 training epoch 시작할 때 호출"""
    #     self.current_epoch = epoch



    def loss(self,
             batch_inputs: Tensor,                            # batch_inputs: 입력 이미지   torch.Size([32, 3, 320, 320])
             batch_data_samples: SampleList) -> SampleList:   # batch_data_samples: 주석 (이미지 정보, 라벨)    dict_keys(['bboxes_labels', 'gazes_labels', 'img_metas'])
        
        # # Hook에서 설정된 current_epoch 사용
        # if (self.current_epoch > self.T1) & (self.current_epoch < self.T2):
        #     gaze_weight = self.lam * (self.current_epoch - self.T1) / (self.T2 - self.T1)
        # elif self.current_epoch >= self.T2:
        #     gaze_weight = self.lam
        # else:
        #     gaze_weight = self.lam / (self.T2 - self.T1)

        neck_x = self.extract_feat(batch_inputs)                 # x.shape = torch.Size([32, 128, 40, 40]), torch.Size([32, 256, 20, 20]), torch.Size([32, 512, 10, 10])
        bbox_losses, results_list = self.bbox_head.loss(neck_x, batch_data_samples)    # yolov5.head.py>def loss에서 반환.
        # with torch.no_grad():
        #     results_list = self.bbox_head.loss(neck_x, batch_data_samples) 
        
        # # label이 input으로 들어오는 경우 실험
        # results_list = batch_data_samples['bboxes_labels']

        l_xyz, l_p3p4p5, l_heatmap, l_heatmap_p3p4p5 = self.gaze_head.loss(neck_x, results_list, batch_data_samples)
        # l_xyz, l_p3p4p5 = self.gaze_head.loss(neck_x, results_list, batch_data_samples)
        # l_xyz, l_heatmap = self.gaze_head.loss(neck_x, results_list, batch_data_samples)
        # l_xyz, l_heatmap = self.gaze_head.loss(neck_x[0], results_list, batch_data_samples)
        # l_pinball, l_pinball_p3p4p5 = self.gaze_head.loss(neck_x, results_list, batch_data_samples)

        # # front_back_classifier 추가
        # l_front_back = self.front_back_classifier_head.loss(neck_x, batch_data_samples)

        total_losses = bbox_losses.copy()        # 'loss_cls','loss_bbox','loss_dfl' 복사
        # total_losses = {}

        # conf 8 실험
        total_losses['loss_xyz'] = l_xyz
        total_losses['loss_p3p4p5'] = l_p3p4p5 
        total_losses['loss_heatmap'] = l_heatmap
        total_losses['loss_heatmap_p3p4p5'] = l_heatmap_p3p4p5

        return total_losses
        
    def predict(self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList,
            rescale: bool = True) -> SampleList:
        """
            추론 단계는 bbox head에서 객체 검출하고, gaze head에서 시선 추적을 동시에 수행함
        """
        
        neck_x = self.extract_feat(batch_inputs)  # backbone, neck에서 feature map 추출
        results_list = self.bbox_head.predict(neck_x, batch_data_samples, rescale=rescale)
        
        gaze_results_list = self.gaze_head.predict(neck_x, results_list, batch_data_samples) 
        # gaze_results_list = self.gaze_head.predict(neck_x[0], results_list, batch_data_samples)  # results_list 들어감.
        
        combined_results = []
        # 두 데이터 합쳐서 하나의 InstanceData 형태로 만들기
        for i in range(len(results_list)):
            scores = results_list[i].scores
            labels = results_list[i].labels
            bboxes = results_list[i].bboxes           
            gazes = gaze_results_list[i].gazes

            if len(bboxes) == 0:
                combined_data = InstanceData(scores=scores, labels=labels, bboxes=bboxes, gazes=gazes)
            else:
                gazes = gazes.repeat(len(bboxes), 1)       # gaze를 복제하여 bboxes와 같은 개수로 확장
                combined_data = InstanceData(scores=scores, labels=labels, bboxes=bboxes, gazes=gazes)
                
            combined_results.append(combined_data)
                   
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, combined_results) 
                       
        return batch_data_samples
    
    # # 여길 안옴. BaseDetector>forward 함수로 감.
    # def _forward(self, 
    #             batch_inputs: Tensor, 
    #             batch_data_samples: SampleList):
    #     """
    #         학습 단계에서 loss계산을 위한 feature 추출을 수행함

    #         디버깅하면 여기로 안옴.
    #     """

    #     neck_x = self.extract_feat(batch_inputs)
    #     outs = self.bbox_head.forward(neck_x)
    #     results_list = self.bbox_head.extract_proposal_feat(*outs)
    #     outs1 = self.gaze_head.forward(neck_x, results_list, batch_data_samples, training=True)
    #     gaze_tensor = outs1[0].gazes

    #     return gaze_tensor
    