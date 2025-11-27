# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmyolo.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors import BaseDetector

from mmengine.structures import InstanceData
from mmyolo.models.gaze_v0.box_filter_layer import FilteringLayer

@MODELS.register_module()
class BisnetGazeDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                #  backbone: ConfigType,
                #  neck: OptConfigType = None,
                #  bbox_head: OptConfigType = None,
                 seg_model: OptConfigType = None, # 추가
                 seg_loss: OptConfigType = None, # 추가
                 gaze_head: ConfigType = None,  # 추가
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # self.backbone = MODELS.build(backbone)
        # if neck is not None:
            # self.neck = MODELS.build(neck)
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        # self.bbox_head = MODELS.build(bbox_head)

        self.seg_model = MODELS.build(seg_model) # 추가
        self.seg_loss = MODELS.build(seg_loss) # 추가
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        gaze_head.update(train_cfg=train_cfg)
        gaze_head.update(test_cfg=test_cfg)
        self.gaze_head = MODELS.build(gaze_head)

        self.box_filter_layer = FilteringLayer(cfg=test_cfg)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        feat_fuse, feat_out_c, feat_out = self.extract_feat(batch_inputs)
        l_seg = self.seg_loss(feat_out, batch_data_samples['seg_maps_path'])

        l_xyz, l_heatmap = self.gaze_head.loss(feat_fuse, feat_out_c, batch_data_samples)

        total_losses = {}
        total_losses['loss_seg'] = l_seg
        total_losses['loss_xyz'] = l_xyz
        total_losses['loss_heatmap'] = l_heatmap  
        #####
        
        return total_losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        feat_fuse, feat_out_c, feat_out = self.extract_feat(batch_inputs)
        # results_list = self.bbox_head.predict(
        #     neck_x, batch_data_samples, rescale=rescale)
        
        # gaze_results_list = self.gaze_head.predict(neck_x[0], results_list, batch_data_samples) 
        
        # combined_results = []
        # # 두 데이터 합쳐서 하나의 InstanceData 형태로 만들기
        # for i in range(len(results_list)):
        #     scores = results_list[i].scores
        #     labels = results_list[i].labels
        #     bboxes = results_list[i].bboxes           
        #     gazes = gaze_results_list[i].gazes

        #     if len(bboxes) == 0:
        #         combined_data = InstanceData(scores=scores, labels=labels, bboxes=bboxes, gazes=gazes)
        #     else:
        #         gazes = gazes.repeat(len(bboxes), 1)       # gaze를 복제하여 bboxes와 같은 개수로 확장
        #         combined_data = InstanceData(scores=scores, labels=labels, bboxes=bboxes, gazes=gazes)
                
        #     combined_results.append(combined_data)

        # batch_data_samples = self.add_pred_to_datasample(
        #     batch_data_samples, combined_results)
        
        gaze_results_list = self.gaze_head.predict(feat_fuse, feat_out_c, batch_data_samples)
        combined_results = []
        for i in range(len(batch_data_samples)):
            gazes = gaze_results_list[i].gazes

            combined_data = InstanceData(gazes=gazes)
            combined_results.append(combined_data)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, combined_results)
        ######
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        # results = self.bbox_head.forward(x)
        results = self.seg_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        # feat_res8, feat_cp8, feat_cp16 = self.backbone(batch_inputs) # feat_res8: [32, 128, 40, 40], feat_cp8: [32, 128, 40, 40], feat_cp16: [32, 128, 20, 20]
        # if self.with_neck:
        #     feat_fuse = self.neck(feat_res8, feat_cp8)    # feat_fuse: [32, 256, 40, 40]
        feat_fuse, feat_out_c, feat_out = self.seg_model(batch_inputs)
        return feat_fuse, feat_out_c, feat_out    # feat_out_c: [16, 19, 64, 64] / [feat_out: [16, 19, 512, 512], feat_out16: [16, 19, 512, 512], feat_out32: [16, 19, 512, 512]]
