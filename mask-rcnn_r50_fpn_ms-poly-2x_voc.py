_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',  # Pascal VOC 转 COCO 格式
    '../_base_/schedules/schedule_2x.py',   # 36 epochs
    '../_base_/default_runtime.py'
]

# VOC 类别
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

# 数据集路径配置（COCO格式的 Pascal VOC）
DATASET_TYPE = 'CocoDataset'
DATA_ROOT = 'data/coco/'

# 数据加载器
train_dataloader = dict(
    batch_size=2,  # 推荐设置以防 OOM
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='RandomResize',
                scale=[(1333, 480), (1333, 960)],
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# 模型头设置 VOC 类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

# 优化器设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# 学习率调度策略：MS + warmup
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True, milestones=[28, 33], gamma=0.1)
]

# 训练流程
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 日志设置
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=10)
)

log_processor = dict(
    type='LogProcessor',
    by_epoch=True,
    window_size=50
)

# TensorBoard 可视化
vis_backends = [
    dict(type='TensorboardVisBackend', save_dir='maskrcnn_results/curves')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 工作目录
work_dir = 'work_dirs_maskrcnn/mask_rcnn_r50_fpn_ms-poly-2x_voc'
