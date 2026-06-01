_base_ = 'yolov8_m_syncbn_fast_8xb16-500e_coco.py'
# _base_ = "yolov8_l_syncbn_fast_8xb16-500e_coco.py"

# data_root = '/mnt/terror/gaze360_dataset/MPIIFaceGaze/annotations/'
data_root = '/mnt/terror/gaze360_dataset/MPIIFaceGaze_v2/Label/'

class_name = ('face','eye')
# class_name = ('face',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60), (220, 20, 60)])
# metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# The output channel of the last stage
last_stage_out_channels = 1024
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config



# max_epochs = 150
max_epochs = 100  # Maximum training epochs
train_batch_size_per_gpu = 32
val_batch_size_per_gpu = 32
train_num_workers = 4
img_scale = (320, 320)

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    # score_thr=0.00,  # NMS+Filter 처리 제거 실험
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300,
    img_scale = (320, 320))  # Max number of detections of each image
# -- ---train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100

# # pre-trained weight를 불러오는 거. load_from에 덮어씌우기
load_from = 'configs/yolov8/mm_yolov8m_coco_state_dict.pth'

# image_path = '/mnt/terror/gaze360_dataset/MPIIFaceGaze/normalized_faces/'
image_path = '/mnt/terror/gaze360_dataset/MPIIFaceGaze_v2/Image/'

model = dict(
    type = 'GazeDetector_v0',                     # 새롭게 정의
    # backbone=dict(frozen_stages=4),          # 백본 모두 얼림.
    # neck = dict(freeze_all=True),            # 넥 얼림.

    bbox_head=dict(head_module=dict(num_classes=num_classes)),    # binary_det 실험할 때 주석처리 
    # GazeHead3 추가
    gaze_head=dict(
        type='GazeHead_v0',
        gaze_head_module=dict(
            type='GazeHeadModule_v0',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        loss_gaze=dict(
            type='GazeLoss_v0',
            loss_weight=6.0,
            batch_size=train_batch_size_per_gpu
        )),

    train_cfg=dict(
        assigner=dict(num_classes=num_classes),       # binary_det 실험할 때 주석처리 
        img_scale = img_scale,
        train_batch = train_batch_size_per_gpu
        ),
    test_cfg=model_test_cfg
    )

# train_pipeline-----------------------
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
last_transform = [
    # dict(type='Float32ToUint8'),  # float32를 uint8로 변환
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
    dict(type='mmdet.RandomFlip', prob=0.5),                             # 좌우반전 추가.
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'seg_map_path',   # seg_map_path 추가
                   'flip_direction'))
]
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    # dict(type='LoadImageFromFile', backend_args=_base_.backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_gaze=True),          # gt_gazes 추가
]

train_pipeline = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
        
    *last_transform
]
# -----------------------------------------

# test_pipeline-----------------------
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    # dict(type='LoadImageFromFile', backend_args=_base_.backend_args, to_float32=True),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LoadAnnotations', with_bbox=True, with_gaze=True, _scope_='mmdet'),   # gt_gazes 추가
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'seg_map_path',   # seg_map_path 추가
                   'scale_factor', 'pad_param'))
]
# -----------------------------------------

# 데이터 로더 설정
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # ann_file='train_v1/mpiifacegaze_p10.json',
        ann_file='train/mpiifacegaze_v2_p10.json',
        data_prefix=dict(img=image_path),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        # ann_file='test_v1/mpiifacegaze_p10.json',
        ann_file='test/mpiifacegaze_v2_p10.json',
        data_prefix=dict(img=image_path),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        # ann_file='test_v1/mpiifacegaze_p10.json',
        ann_file='test/mpiifacegaze_v2_p10.json',
        data_prefix=dict(img=image_path),
        pipeline=test_pipeline))

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = [
    dict(
        type='AngularError',  # AngularError metric
        # ann_file=data_root + 'test_v1/mpiifacegaze_p10.json'
        ann_file=data_root + 'test/mpiifacegaze_v2_p10.json'
    ),
    dict(
    type='mmdet.CocoMetric',
    classwise = True,            # 카테고리 별 AP 계산
    proposal_nums=(100, 1, 10),  # COCO metric에 대한 proposal 숫자
    # ann_file=data_root + 'test_v1/mpiifacegaze_p10.json',
    ann_file=data_root + 'test/mpiifacegaze_v2_p10.json',
    metric='bbox'  # bbox 성능 평가
    )
]
test_evaluator = [dict(
    type='mmdet.CocoMetric',
    classwise = True,            # 카테고리 별 AP 계산
    proposal_nums=(100, 1, 10),  # COCO metric에 대한 proposal 숫자
    # ann_file=data_root + 'test_v1/mpiifacegaze_p10.json',
    ann_file=data_root + 'test/mpiifacegaze_v2_p10.json',
    metric='bbox'  # bbox 성능 평가
    ),
    dict(
        type='AngularError',  # AngularError metric
        # ann_file=data_root + 'test_v1/mpiifacegaze_p10.json'
        ann_file=data_root + 'test/mpiifacegaze_v2_p10.json'
    )
]

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, less_keys=['MAE_360'], save_best='auto'),  # less_keys 추가. less_keys가 None이면 _default_less_keys 사용.
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=5,  # 성능 향상이 없을 때 몇 에포크까지 기다릴지
        min_delta=0.05,
        monitor='MAE_360',  # 어떤 지표를 기준으로 성능을 모니터링할지 (rule이 없을 때 비교지표라서 monitor 사용 안함)
        rule='less'
    )
)


# custom_hooks = [
#     dict(type='EpochUpdateHook')
# ]

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa