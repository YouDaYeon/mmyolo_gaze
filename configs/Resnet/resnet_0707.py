_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/mnt/terror/gaze360_dataset/gaze_annotations/'
image_path = '/mnt/terror/gaze360_dataset/original/imgs/'

# # RT-GENE 데이터
# data_root = '/mnt/terror/gaze360_dataset/RT-GENE/'
# image_path = "/mnt/terror/gaze360_dataset/RT-GENE/2529036"

class_name = ('face','eye')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60), (220, 20, 60)])

# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 4
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.001
max_epochs = 100  # Maximum training epochs

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    # score_thr=0.00,  # NMS+Filter 처리 제거 실험
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300,
    img_scale = (512, 512))  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (512, 512)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# # Strides of multi-scale prior box
strides = [8, 16, 32]
# The output channel of the last stage
last_stage_out_channels = 1024
# num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -- ---train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validationintervals in stage 1
save_epoch_intervals = 1
# validation intervals in stage 2
val_interval_stage2 = 1

# The maximum checkpoints to keep.
max_keep_ckpts = 2
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# pre-trained weight를 불러오는 거. load_from에 덮어씌우기
# load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa
# load_from = 'work_dirs3/yolov8_s_onlygaze/rt_gene_train/best_MAE_360_epoch_9.pth'  # noqa
# load_from = 'configs/Resnet/mm_resnet34.pth'
load_from = 'configs/Resnet/mm_resnet34_seg_head2.pth'

# ===============================Unmodified in most cases====================
model = dict(
    type='BisnetGazeDetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],   # 각 픽셀 값 [0,255]->[0,1] 로 변환. 세 개의 체널 RGB 모두 동일하게 적용.
        bgr_to_rgb=True),
    seg_model = dict(
        type='BiSeNet',
        num_classes=19,
        backbone_name="resnet34"
    ),
    seg_loss = dict(
        type='OhemLossWrapper',
        thresh = 0.7,
        min_kept = train_batch_size_per_gpu * img_scale[0] * img_scale[1] // 16,  # 262144
        loss_weight=6.0,
    ),
    # backbone=dict(
    #     type='ContextPath',
    #     backbone_name="resnet34",
    #     frozen_stages=4,  # freeze 설정에 따라 결정
    #     ),
    # neck = dict(
    #     type='FeatureFusionModule',
    #     in_channels=256,
    #     out_channels=256,
    #     frozen=True,  # freeze 설정에 따라 결정
    # ),
    # seg_head=dict(
    #     type='BiSeNetOutput',
    #     in_channels=[256, 128, 128],  # [feat_fuse, feat_cp8, feat_cp16] 채널 수
    #     mid_channels=[256, 64, 64],   # 각 스케일별 중간 채널 수
    #     num_classes=19,
    #     frozen=True,  # freeze 설정에 따라 결정
    # ),
    # bbox_head=dict(
    #     type='YOLOv8Head',
    #     head_module=dict(
    #         type='YOLOv8HeadModule',
    #         num_classes=num_classes,
    #         # in_channels=[256, 512, last_stage_out_channels],
    #         in_channels=[512, 256, 256],
    #         widen_factor=widen_factor,
    #         reg_max=16,
    #         norm_cfg=norm_cfg,
    #         act_cfg=dict(type='SiLU', inplace=True),
    #         featmap_strides=strides),
    #     prior_generator=dict(
    #         type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
    #     bbox_coder=dict(type='DistancePointBBoxCoder'),
    #     # scaled based on number of detection layers
    #     loss_cls=dict(
    #         type='mmdet.CrossEntropyLoss',
    #         use_sigmoid=True,
    #         reduction='none',
    #         loss_weight=loss_cls_weight),
    #     loss_bbox=dict(
    #         type='IoULoss',
    #         iou_mode='ciou',
    #         bbox_format='xyxy',
    #         reduction='sum',
    #         loss_weight=loss_bbox_weight,
    #         return_iou=False),
    #     loss_dfl=dict(
    #         type='mmdet.DistributionFocalLoss',
    #         reduction='mean',
    #         loss_weight=loss_dfl_weight)),
    # GazeHead 추가
    gaze_head=dict(
        type='BisnetGazeHead_v0',
        gaze_head_module=dict(
            type='BisnetGazeHeadModule_v0',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),

        loss_gaze=dict(
            type='GazeLoss_v0',
            loss_weight=6.0,
            batch_size=train_batch_size_per_gpu
        )),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9),
        img_scale = img_scale,
        train_batch = train_batch_size_per_gpu),
    test_cfg= model_test_cfg)


albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_gaze=True)          # gt_gazes 추가
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            'gt_gazes': 'gazes'                                          # gaze 정보
        }),
    
    dict(type='YOLOv5HSVRandomAug'),                                     # 이미지의 색조, 채도, 밝기 조정
    # dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'seg_map_path',   # seg_map_path 추가
                   'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    # dict(
    #     type='Mosaic',
    #     img_scale=img_scale,
    #     pad_val=114.0,
    #     pre_transform=pre_transform),
    *last_transform
]
# test_pipeline-----------------------
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    # dict(
    #     type='LetterResize',
    #     scale=img_scale,
    #     allow_scale_up=False,
    #     pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, with_gaze=True, _scope_='mmdet'),   # gt_gazes 추가
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'seg_map_path',   # seg_map_path 추가
                   'scale_factor', 'pad_param'))
]
# -----------------------------------------

# 데이터 로더 설정
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='gaze_train_all_bbox.json',
        # ann_file='RT-GENE_train.json',
        data_prefix=dict(img=image_path),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=image_path),
        ann_file='gaze_val_all_bbox.json',
        # ann_file='RT-GENE_val.json',
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=image_path),
        ann_file='gaze_test_all_bbox.json',
        # ann_file='RT-GENE_test.json',
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

# _base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = [
    dict(
        type='AngularError',  # AngularError metric
        ann_file=data_root + 'gaze_val_all_bbox.json'
        # ann_file=data_root + 'RT-GENE_val.json'
    ),
    # dict(
    #     type='mmdet.CocoMetric',
    #     classwise = True,            # 카테고리 별 AP 계산
    #     proposal_nums=(100, 1, 10),  # COCO metric에 대한 proposal 숫자
    #     ann_file=data_root + 'gaze_val_all_bbox.json',  
    #     metric='bbox'  # bbox 성능 평가
    # )
]
test_evaluator = [
    # dict(
    #     type='mmdet.CocoMetric',
    #     classwise = True,            # 카테고리 별 AP 계산
    #     proposal_nums=(100, 1, 10),  # COCO metric에 대한 proposal 숫자
    #     ann_file=data_root + 'gaze_test_all_bbox.json',  
    #     metric='bbox'  # bbox 성능 평가
    # ),
    dict(
        type='AngularError',  # AngularError metric
        ann_file=data_root + 'gaze_test_all_bbox.json'
        # ann_file=data_root + 'RT-GENE_test.json'
    )
]

param_scheduler = None
# optim_wrapper = dict(
#     type='OptimWrapper',
#     clip_grad=dict(max_norm=10.0),
#     optimizer=dict(
#         type='SGD',
#         lr=base_lr,
#         momentum=0.937,
#         weight_decay=weight_decay,
#         nesterov=True,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#     constructor='YOLOv5OptimizerConstructor')

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-08,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs,
        warmup_mim_iter=10),
    checkpoint=dict(
        type='CheckpointHook',
        # interval=10,
        interval=1,
        save_best='auto',
        # save_best=None,
        max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=5),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=5,  # 성능 향상이 없을 때 몇 에포크까지 기다릴지
        min_delta=0.05,
        monitor='MAE_360',  # 어떤 지표를 기준으로 성능을 모니터링할지
        rule='less'
    )
    )

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,         # 최대 에폭 수
    val_interval=1)                # Validation 간격 (몇 번째 에폭마다 validation을 할 것인가)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
