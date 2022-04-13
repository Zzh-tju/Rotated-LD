# Copyright (c) OpenMMLab. All rights reserved.
from .base import RotatedBaseDetector
from .gliding_vertex import GlidingVertex
from .oriented_rcnn import OrientedRCNN
from .r3det import R3Det
from .redet import ReDet
from .roi_transformer import RoITransformer
from .rotate_faster_rcnn import RotatedFasterRCNN
from .rotated_reppoints import RotatedRepPoints
from .rotated_retinanet import RotatedRetinaNet
from .rotated_retinanet_distribution import RotatedRetinaNetDistribution
from .s2anet import S2ANet
from .single_stage import RotatedSingleStageDetector
from .two_stage import RotatedTwoStageDetector
from .rotated_kd_one_stage import KnowledgeDistillationRotatedSingleStageDetector

__all__ = [
    'RotatedRetinaNet', 'RotatedRetinaNetDistribution', 'RotatedFasterRCNN', 'OrientedRCNN', 'RoITransformer',
    'GlidingVertex', 'ReDet', 'R3Det', 'S2ANet', 'RotatedRepPoints',
    'RotatedBaseDetector', 'RotatedTwoStageDetector',
    'RotatedSingleStageDetector', 'KnowledgeDistillationRotatedSingleStageDetector'
]
