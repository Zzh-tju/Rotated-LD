from re import S
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def im_loss(x, soft_target):
    # print(x.shape, soft_target.shape)
    # print(F.mse_loss(x, soft_target))
    return F.mse_loss(x, soft_target)

@ROTATED_LOSSES.register_module()
class IMLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                x,
                soft_target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_im = self.loss_weight * im_loss(
            x, soft_target, reduction=reduction)

        return loss_im

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def relation_loss1(feature_s, feature_t):
    """Compute loss of a single scale level.
    Args:
        feature_s: student's feature map	# N*C*H*W
        feature_t: teacher's feature map with the same size as feature_s
    Returns:
        Loss of relation distillation
    """
    N,C,H,W=feature_s.size()
    s = feature_s.reshape(-1, H*W)
    t = feature_t.reshape(-1, H*W)

    matrix_s1=s.unsqueeze(1).expand(N*C,H*W,H*W)
    matrix_s2=s.unsqueeze(2).expand(N*C,H*W,H*W)
    relation_matrix_s=((matrix_s1-matrix_s2)**2)/((H*W)**2)

    matrix_t1=t.unsqueeze(1).expand(N*C,H*W,H*W)
    matrix_t2=t.unsqueeze(2).expand(N*C,H*W,H*W)
    relation_matrix_t=((matrix_t1-matrix_t2)**2)/((H*W)**2)

    return F.mse_loss(relation_matrix_s, relation_matrix_t)


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def relation_loss(feature_s, feature_t):
    """Compute loss of a single scale level.
    Args:
        feature_s: student's feature C*N
        feature_t: teacher's feature map with the same size as feature_s
    Returns:
        Loss of relation distillation
    """
    C,N=feature_s.size()

    matrix_s1=feature_s.unsqueeze(1).expand(C,N,N)
    matrix_s2=feature_s.unsqueeze(2).expand(C,N,N)
    relation_matrix_s=(matrix_s1-matrix_s2)**2
    #with torch.no_grad():
        #u_s = relation_matrix_s.sum(dim=1).sum(dim=1).unsqueeze(1).unsqueeze(1).expand(C,N,N)

    matrix_t1=feature_t.unsqueeze(1).expand(C,N,N)
    matrix_t2=feature_t.unsqueeze(2).expand(C,N,N)
    relation_matrix_t=(matrix_t1-matrix_t2)**2
    #u_t = relation_matrix_t.sum(dim=1).sum(dim=1).unsqueeze(1).unsqueeze(1).expand(C,N,N)
    #print('---------------------------')
    #print(F.mse_loss(relation_matrix_s/u_s, relation_matrix_t/u_t))
    return F.mse_loss(relation_matrix_s, relation_matrix_t)

@ROTATED_LOSSES.register_module()
class RelationLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                x,
                soft_target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_relation = self.loss_weight * relation_loss(
            x, soft_target, reduction=reduction)

        return loss_relation

