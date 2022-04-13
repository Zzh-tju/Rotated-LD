_base_ = ['../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py']


teacher_ckpt = 'r50-gwd-1x.pth'
model = dict(
    type='KnowledgeDistillationRotatedSingleStageDetector',
    teacher_config='configs/gwd/rotated_retinanet_distribution_hbb_gwd_r50_fpn_1x_dota_oc.py',
    teacher_ckpt=teacher_ckpt,
    output_feature=True,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='LDRotatedRetinaHead',
        reg_max=8,
        reg_decoded_bbox=True,
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=30, T=5),
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=30, T=5),
        loss_im=dict(type='IMLoss', loss_weight=2.0),
        imitation_method='finegrained',  # gibox, finegrain, decouple, fitnet
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(train=dict(pipeline=train_pipeline))
