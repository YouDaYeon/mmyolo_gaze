# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import numpy as np
import torch
from mmengine.dataset import COLLATE_FUNCTIONS
from mmengine.dist import get_dist_info

from ..registry import TASK_UTILS


@COLLATE_FUNCTIONS.register_module()
def yolov5_collate(data_batch: Sequence,
                   use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_gaze_labels = []
    batch_masks = []
    batch_keyponits = []
    batch_keypoints_visible = []

    ori_shapes = [] #########################################################
    imgs_path = [] ###########################################################
    # seg_maps_path = [] #########################################################

    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        gt_gazes = datasamples.gt_instances.gazes                 # gt_gazes 가져오기

        # 추가
        ori_shape = datasamples.ori_shape ####################################
        ori_shapes.append(ori_shape) ########################################
        img_path = datasamples.img_path ########################################
        imgs_path.append(img_path) ########################################

        # seg_map_path = datasamples.seg_map_path ########################################
        # seg_maps_path.append(seg_map_path) ########################################

        # # gt_gazes가 비어 있는 경우 (행이 0인 경우)에도 (0, 3) 형식으로 맞춰준다.
        # if gt_gazes.numel() == 0:  # numel()은 텐서의 요소 수를 반환한다.
        #     gt_gazes = torch.empty(0, 3).to(gt_gazes.device)  # 비어 있는 텐서를 (0, 3)으로 정의

        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks
            batch_masks.append(masks)
        if 'gt_panoptic_seg' in datasamples:
            batch_masks.append(datasamples.gt_panoptic_seg.pan_seg)
        if 'keypoints' in datasamples.gt_instances:
            keypoints = datasamples.gt_instances.keypoints
            keypoints_visible = datasamples.gt_instances.keypoints_visible
            batch_keyponits.append(keypoints)
            batch_keypoints_visible.append(keypoints_visible)
    
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

        # gazes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_gazes),
        #                           dim=1)
        gazes_labels = gt_gazes
        batch_gaze_labels.append(gazes_labels)

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        },
        'gaze_samples': {
            'gazes_labels' : torch.cat(batch_gaze_labels, 0)
        },
        'ori_shapes': {
            'ori_shapes': ori_shapes
        },
        'imgs_path': {
            'imgs_path': imgs_path
        },
        # 'seg_maps_path': {
        #     'seg_maps_path': seg_maps_path
        # }
    }

    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if len(batch_keyponits) > 0:
        collated_results['data_samples']['keypoints'] = torch.cat(
            batch_keyponits, 0)
        collated_results['data_samples']['keypoints_visible'] = torch.cat(
            batch_keypoints_visible, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)
    return collated_results


@TASK_UTILS.register_module()
class BatchShapePolicy:
    """BatchShapePolicy is only used in the testing phase, which can reduce the
    number of pad pixels during batch inference.

    Args:
       batch_size (int): Single GPU batch size during batch inference.
           Defaults to 32.
       img_size (int): Expected output image size. Defaults to 640.
       size_divisor (int): The minimum size that is divisible
           by size_divisor. Defaults to 32.
       extra_pad_ratio (float):  Extra pad ratio. Defaults to 0.5.
    """

    def __init__(self,
                 batch_size: int = 32,
                 img_size: int = 640,
                 size_divisor: int = 32,
                 extra_pad_ratio: float = 0.5):
        self.img_size = img_size
        self.size_divisor = size_divisor
        self.extra_pad_ratio = extra_pad_ratio
        _, world_size = get_dist_info()
        # During multi-gpu testing, the batchsize should be multiplied by
        # worldsize, so that the number of batches can be calculated correctly.
        # The index of batches will affect the calculation of batch shape.
        self.batch_size = batch_size * world_size

    def __call__(self, data_list: List[dict]) -> List[dict]:
        image_shapes = []
        for data_info in data_list:
            image_shapes.append((data_info['width'], data_info['height']))

        image_shapes = np.array(image_shapes, dtype=np.float64)

        n = len(image_shapes)  # number of images
        batch_index = np.floor(np.arange(n) / self.batch_size).astype(
            np.int64)  # batch index
        number_of_batches = batch_index[-1] + 1  # number of batches

        aspect_ratio = image_shapes[:, 1] / image_shapes[:, 0]  # aspect ratio
        irect = aspect_ratio.argsort()

        data_list = [data_list[i] for i in irect]

        aspect_ratio = aspect_ratio[irect]
        # Set training image shapes
        shapes = [[1, 1]] * number_of_batches
        for i in range(number_of_batches):
            aspect_ratio_index = aspect_ratio[batch_index == i]
            min_index, max_index = aspect_ratio_index.min(
            ), aspect_ratio_index.max()
            if max_index < 1:
                shapes[i] = [max_index, 1]
            elif min_index > 1:
                shapes[i] = [1, 1 / min_index]

        batch_shapes = np.ceil(
            np.array(shapes) * self.img_size / self.size_divisor +
            self.extra_pad_ratio).astype(np.int64) * self.size_divisor

        for i, data_info in enumerate(data_list):
            data_info['batch_shape'] = batch_shapes[batch_index[i]]

        return data_list
