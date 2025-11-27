_base_ = 'yolov8_m_syncbn_fast_8xb16-500e_coco.py'
# _base_ = "yolov8_l_syncbn_fast_8xb16-500e_coco.py"

data_root = '/mnt/terror/gaze360_dataset/gaze_annotations/'
# data_root = '/mnt/terror/gaze360_dataset/gaze_360_180/'
# data_root = '/mnt/terror/gaze360_dataset/gaze_360_refine_body/'

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
# img_scale = (640, 640)
# img_scale = (960, 960)
# img_scale = (256, 768)
# img_scale = (320, 640)
# img_scale = (128, 384)

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
    # img_scale = (640, 640))  # Max number of detections of each image
    # img_scale = (960, 960))  # Max number of detections of each image
    # img_scale = (256, 768))  # Max number of detections of each image
    # img_scale = (320, 640))  # Max number of detections of each image
    # img_scale = (128, 384))  # Max number of detections of each image

# -- ---train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100

# # pre-trained weight를 불러오는 거. load_from에 덮어씌우기
load_from = 'configs/yolov8/mm_yolov8m_coco_state_dict.pth'
# load_from = 'pretrined_weight/mm_yolov8m-face_coco.pth'
# load_from = '/mnt/terror/gaze360_dataset/best_coco_bbox_mAP_epoch_39.pth'
# load_from = '/mnt/terror/gaze360_dataset/gaze360_0702_best.pth'
# load_from = '/mnt/terror/gaze360_dataset/best_coco_face_precision_epoch_22.pth'
# load_from = '/mnt/terror/gaze360_dataset/mm_yolov8l_coco.pth'

image_path = '/mnt/terror/gaze360_dataset/original/imgs/'
# image_path = '/mnt/terror/gaze360_dataset/gaze_360_180/Image/'
# image_path = '/mnt/terror/gaze360_dataset/'

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

    # # front_back_classifier 추가
    # front_back_classifier_head = dict(
    #     type='FrontBackClassifierHead_v0',
    #     loss_front_back=dict(
    #         type='FrontBackLoss_v0',
    #         loss_weight=3.0,
    #         batch_size=train_batch_size_per_gpu
    #     )
    # ),

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
    dict(type='CLAHE', p=0.01),
    # dict(type='GridDropout', ratio=0.5, unit_size_min=10, unit_size_max=20, random_offset=True, p=0.5)  # GridDropout 추가

    # dict(type='GaussNoise', var_limit=(0.001, 0.01), p=0.5),  # float32 이미지 기준
    # dict(type='GaussNoise', var_limit=(0.001, 0.01), p=0.01),  # float32 이미지 기준
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

    # dict(
    #     type='mmdet.RandomCrop',
    #     crop_type='relative_range',         # 비율 기반 랜덤 범위 crop
    #     crop_size=(0.4, 0.4),               # 최소 40%, 최대 100%까지 랜덤 (w, h)
    #     allow_negative_crop=False,          # bbox가 하나도 안 남으면 버림
    #     bbox_clip_border=True               # bbox가 이미지 밖으로 나가면 잘라냄
    # ),  

    # dict(type='Resize', scale=img_scale, keep_ratio=False),  # random crop 뒤에 resize 추가

    # dict(
    #     type='CircularShift',
    #     axis='horizontal',      # or 'vertical'
    #     num_parts=7,
    #     prob=0.5
    # ),

    # dict(
    #     type='mmdet.RandomErasing',
    #     prob=0.5,                     # ***** 50% 확률 적용 ***** <- 기본 항상 적용이라 클래스 내부 수정.
    #     n_patches=1,                  # 1개 패치 적용
    #     ratio=(0.02, 0.2),            # sl = 0.02, sh = 0.2 면적 비율: 전체 이미지의 2~20%
    #     squared=False,                # 다양한 종횡비 허용 (정사각형 아님)
    #     bbox_erased_thr=0.9,          # bbox의 90% 이상 가려지면 제거
    #     img_border_value=0,           # 검정색으로 채움
    #     mask_border_value=0,          # 마스크도 검정색으로 채움
    #     seg_ignore_label=255          # segmentation용 무시값
    # ),


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
    # dict(
    #     type='LetterResize',
    #     scale=img_scale,
    #     allow_scale_up=True,
    #     pad_val=dict(img=114.0)),
    # dict(
    #     type='RandomCenterCropPad',
    #     crop_size=img_scale,
    #     ratios=(0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
    #     border=128,
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True,
    #     test_mode=False,
    #     test_pad_mode=None),
    # dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    # dict(
    #     type='Mosaic',
    #     img_scale=img_scale,
    #     pad_val=114.0,
    #     pre_transform=pre_transform),
    # dict(
    #     type='YOLOv5RandomAffine',
    #     # max_rotate_degree=0.0,                                       # 회전 최대 각도
    #     max_rotate_degree=30.0,
    #     # max_translate_ratio=0.15,            # 15% 이동
    #     max_translate_ratio=0.0,    
    #     max_shear_degree=0.0,                                        # 기울임 최대 각도
    #     # scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),    # 이미지 확대 및 축소 비율 범위. 0.5~1.5는 50% 150% 확대/축소 가능
    #     scaling_ratio_range=(1.0, 1.0),      # 크기 고정
    #     max_aspect_ratio=max_aspect_ratio,                           # 거의 필터링 없음
    #     # img_scale is (width, height)
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2),             # 이미지 변형 후 경계값 설정. 음수값은 경계를 확장하는 방식.
    #     border_val=(114, 114, 114)),            


    *last_transform
]
# -----------------------------------------

# test_pipeline-----------------------
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    # dict(type='LoadImageFromFile', backend_args=_base_.backend_args, to_float32=True),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    # dict(
    #     type='RandomCenterCropPad',
    #     crop_size=None,
    #     ratios=None,
    #     border=None,
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True,
    #     test_mode=True,
    #     test_pad_mode=('logical_or', 127)),
    # dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    # dict(
    #     type='LetterResize',
    #     scale=img_scale,
    #     allow_scale_up=False,
    #     pad_val=dict(img=114)),
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
        ann_file='gaze_train_all_bbox.json',
        # ann_file='gaze_test_1.json',
        # ann_file='refine/gaze360_train_refine_include_noise.json',
        # ann_file='gaze360_train_180_facecrop.json',
        # ann_file = 'gaze360_train_body.json',
        # ann_file = 'gaze360_train_body_refine_include_noise.json',
        data_prefix=dict(img=image_path),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='gaze_val_all_bbox.json',
        # ann_file='gaze_test_1.json',
        # ann_file='refine/gaze360_val_refine_include_noise.json',
        # ann_file='gaze360_val_180_facecrop.json',
        # ann_file = 'gaze360_val_body.json',
        # ann_file = 'gaze360_val_body_refine_include_noise.json',
        data_prefix=dict(img=image_path),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='gaze_test_all_bbox.json',
        # ann_file='gaze360_test_180.json',
        # ann_file='gaze360_test_noise.json',
        # ann_file='gaze360_test_body.json',
        # ann_file='refine/gaze360_val_refine.json',
        # ann_file='gaze360_test_180_facecrop.json',
        data_prefix=dict(img=image_path),
        pipeline=test_pipeline))

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
# _base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

# # 얼린 레이어와 학습 가능한 레이어에 다른 학습률 적용
# _base_.optim_wrapper.optimizer = dict(
#     type='SGD',
#     lr=0.01,  # 기본 학습률
#     momentum=0.937,
#     weight_decay=0.0005,
#     nesterov=True,
#     batch_size_per_gpu=train_batch_size_per_gpu
# )

# # 파라미터 그룹별 학습률 설정
# _base_.optim_wrapper.paramwise_cfg = dict(
#     base_total_batch_size=32,  # 실제 배치 크기와 맞춤 (32 * 1 GPU)
#     custom_keys={
#         'backbone': dict(lr_mult=0.0),  # 얼린 레이어는 학습률 0
#         'neck': dict(lr_mult=0.0),      # 얼린 레이어는 학습률 0
#         'bbox_head': dict(lr_mult=0.0), # 얼린 레이어는 학습률 0
#         'gaze_head': dict(lr_mult=1.0), # gaze_head는 정상 학습률
#     }
# )

# val_evaluator = dict(
#                     type = 'AngularError',
#                     ann_file=data_root + 'gaze_val_all_bbox.json')
# test_evaluator = dict(
#                     type = 'AngularError',
#                     ann_file=data_root + 'gaze_test_all_bbox.json')

val_evaluator = [
    dict(
        type='AngularError',  # AngularError metric
        ann_file=data_root + 'gaze_val_all_bbox.json'
        # ann_file=data_root + 'gaze_test_1.json'
        # ann_file=data_root + 'refine/gaze360_val_refine_include_noise.json',
        # ann_file=data_root + 'gaze360_val_180_facecrop.json'
        # ann_file=data_root + 'gaze360_val_body.json',
        # ann_file=data_root + 'gaze360_val_body_refine_include_noise.json',
    ),
    dict(
    type='mmdet.CocoMetric',
    classwise = True,            # 카테고리 별 AP 계산
    proposal_nums=(100, 1, 10),  # COCO metric에 대한 proposal 숫자
    ann_file=data_root + 'gaze_val_all_bbox.json',
    # ann_file=data_root + 'gaze_test_1.json',
    # ann_file=data_root + 'refine/gaze360_val_refine_include_noise.json',
    # ann_file=data_root + 'gaze360_val_180_facecrop.json',
    # ann_file=data_root + 'gaze360_val_body.json',
    # ann_file=data_root + 'gaze360_val_body_refine_include_noise.json',
    metric='bbox'  # bbox 성능 평가
    )
]
test_evaluator = [dict(
    type='mmdet.CocoMetric',
    classwise = True,            # 카테고리 별 AP 계산
    proposal_nums=(100, 1, 10),  # COCO metric에 대한 proposal 숫자
    ann_file=data_root + 'gaze_test_all_bbox.json',
    # ann_file=data_root + 'gaze_train_all_bbox.json',
    # ann_file=data_root + 'gaze360_test_180.json',
    # ann_file=data_root + 'gaze360_test_noise.json',
    # ann_file=data_root + 'gaze360_test_body.json',
    # ann_file=data_root + 'refine/gaze360_val_refine.json',
    # ann_file=data_root + 'gaze360_test_180_facecrop.json',
    metric='bbox'  # bbox 성능 평가
    ),
    dict(
        type='AngularError',  # AngularError metric
        ann_file=data_root + 'gaze_test_all_bbox.json'
        # ann_file=data_root + 'gaze_train_all_bbox.json'
        # ann_file=data_root + 'gaze360_test_180.json'
        # ann_file=data_root + 'gaze360_test_noise.json'
        # ann_file=data_root + 'gaze360_test_body.json'
        # ann_file=data_root + 'refine/gaze360_val_refine.json'
        # ann_file=data_root + 'gaze360_test_180_facecrop.json'
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
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa