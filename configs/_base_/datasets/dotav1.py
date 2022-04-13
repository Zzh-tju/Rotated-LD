# dataset settings
dataset_type = 'DOTADataset'
data_root = '/media/ssd/datasets/DOTAv1/split_ss_dota1_0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/media/ssd/datasets/DOTAv1/split_ss_dota1_0/train/annfiles/',
        img_prefix='/media/ssd/datasets/DOTAv1/split_ss_dota1_0/train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/media/ssd/datasets/DOTAv1/split_ss_dota1_0/val/annfiles/',
        img_prefix='/media/ssd/datasets/DOTAv1/split_ss_dota1_0/val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/media/ssd/datasets/DOTAv1/split_ss_dota1_0/val/annfiles/',
        img_prefix='/media/ssd/datasets/DOTAv1/split_ss_dota1_0/val/images/',
        pipeline=test_pipeline))
