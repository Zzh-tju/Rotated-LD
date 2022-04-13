import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from mmrotate.core import (build_assigner, build_bbox_coder, build_prior_generator, 
                        build_sampler, obb2hbb, rotated_anchor_inside_flags)
from mmdet.models.builder import HEADS

from .rotated_retina_head_distribution import RotatedRetinaHeadDistribution
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from ..builder import ROTATED_HEADS, build_loss

@ROTATED_HEADS.register_module()
class LDRotatedRetinaHead(RotatedRetinaHeadDistribution):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_ld=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_im=dict(type='IMLoss', loss_weight=2.0),
                 imitation_method='gibox',
                 **kwargs):
        super(LDRotatedRetinaHead, self).__init__(
            num_classes,
            in_channels,
            **kwargs)
        self.imitation_method = imitation_method
        self.loss_im = build_loss(loss_im)
        self.loss_ld = build_loss(loss_ld)
        self.loss_kd = build_loss(loss_kd)
        self.iou_calculator = build_iou_calculator(
            dict(type='BboxOverlaps2D'), )

    def forward_train(self,
                      x,
                      out_teacher,
                      teacher_x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
        Returns:
            tuple[dict, list]: The loss components and proposals of each image.
            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, out_teacher, x, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, out_teacher, x,
                                  teacher_x, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, soft_targets, soft_labels, x, teacher_x,
                    ld_region, im_region, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
            weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)

        label_weights = label_weights.reshape(-1)
        ld_region = ld_region.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        soft_labels = soft_labels.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # feature imitation loss
        N, C, _, _ = x.size()
        teacher_x = teacher_x.permute(0, 2, 3, 1).reshape(-1, C)
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        if self.imitation_method == 'gibox':
            gi_idx = self.get_gi_region(soft_label, cls_score, anchors,
                                        bbox_pred, soft_targets, stride)
            gi_teacher = teacher_x[gi_idx]
            gi_student = x[gi_idx]

            loss_im = self.loss_im(gi_student, gi_teacher)
        elif self.imitation_method == 'decouple':
            fg_inds = (im_region > 0).nonzero().squeeze(1)
            ng_inds = (im_region == 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[fg_inds],
                                       teacher_x[fg_inds]) + 2 * self.loss_im(
                                           x[ng_inds], teacher_x[fg_inds])
            else:
                loss_im = bbox_pred.sum() * 0
        else:
            assigned_im = ((im_region.reshape(N, -1, int(im_region.size()[1]/9)).max(1)[0])>0).reshape(-1)
            fg_inds = (assigned_im > 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[assigned_im], teacher_x[assigned_im])
            else:
                loss_im = bbox_pred.sum() * 0



        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, (self.reg_max + 1) * 5)
        soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, (self.reg_max + 1) * 5)
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        encode_pred = self.integral(bbox_pred[:,:((self.reg_max + 1) * 4)].reshape(-1, self.reg_max + 1))
        encode_angle_pred = self.integral_angle(bbox_pred[:,((self.reg_max + 1) * 4):].reshape(-1, self.reg_max + 1))
        encode_bbox_pred = torch.stack([encode_pred[:,0], encode_pred[:,1], encode_pred[:,2], encode_pred[:,3], encode_angle_pred], dim=1)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            decode_bbox_pred = self.bbox_coder.decode(anchors, encode_bbox_pred)
            loss_bbox = self.loss_bbox(
                decode_bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)
        else:
            loss_bbox = self.loss_bbox(
                encode_bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)

        ld_num = torch.clamp((ld_region > 0).float().sum(), min=1.0)

        pred_corners = bbox_pred.reshape(-1, self.reg_max + 1)
        soft_corners = soft_targets.reshape(-1, self.reg_max + 1)


        loss_ld = self.loss_ld(
            pred_corners, soft_corners, weight=ld_region[:, None].expand(-1, 5).reshape(-1), avg_factor=ld_num)

        loss_kd = self.loss_kd(cls_score, soft_labels, weight=ld_region, avg_factor=ld_num)
        return loss_cls, loss_bbox, loss_ld, loss_kd, loss_im

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             soft_teacher,
             x,
             teacher_x,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        soft_label, soft_target = soft_teacher
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, ld_region_list, im_region_list) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_ld, losses_kd, losses_im= multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            soft_target,
            soft_label,
            x,
            teacher_x,
            ld_region_list,
            im_region_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_ld=losses_ld,
            loss_kd=losses_kd,
            loss_im=losses_im)


    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors ,4)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if self.assign_by_circumhbbox is not None:
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, gt_bboxes_assign, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        ld_region = self.get_ld_region(anchors, num_level_anchors_inside, gt_bboxes)
        im_region = self.get_im_region(anchors, gt_bboxes, mode=self.imitation_method)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            ld_region = unmap(ld_region, num_total_anchors, inside_flags)
            im_region = unmap(im_region, num_total_anchors, inside_flags)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, ld_region, im_region)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list, all_ld_region, all_im_region) = results[:9]
        rest_results = list(results[9:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        ld_regions_list = images_to_levels(all_ld_region,
                                             num_level_anchors)
        im_regions_list = images_to_levels(all_im_region, num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg, ld_regions_list, im_regions_list)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def get_ld_region(
        self,
        bboxes,
        num_level_bboxes,
        gt_bboxes,
    ):
        """Assign gt to bboxes.
        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt
        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        self.topk = 9
        # compute iou between all bbox and gt

        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
        axmin = bboxes[:, 0] - bboxes[:, 2]/2
        aymin = bboxes[:, 1] - bboxes[:, 3]/2
        axmax = bboxes[:, 0] + bboxes[:, 2]/2
        aymax = bboxes[:, 1] + bboxes[:, 3]/2
        bboxes = torch.stack([axmin, aymin, axmax, aymax], dim=1)

        txmin = gt_bboxes[:, 0] - gt_bboxes[:, 2]/2
        tymin = gt_bboxes[:, 1] - gt_bboxes[:, 3]/2
        txmax = gt_bboxes[:, 0] + gt_bboxes[:, 2]/2
        tymax = gt_bboxes[:, 1] + gt_bboxes[:, 3]/2
        gt_bboxes = torch.stack([txmin, tymin, txmax, tymax], dim=1)

        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        #print('iou',overlaps)
        #diou = self.iou_calculator(bboxes, gt_bboxes, mode='diou')
        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assigned_ld = (assigned_gt_inds + 0).float()

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        candidate_idxs_t = []

        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_t = min(self.topk, bboxes_per_level)
            selectable_k = min(bboxes_per_level, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            _, topt_idxs_per_level = distances_per_level.topk(
                selectable_t, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            candidate_idxs_t.append(topt_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        candidate_idxs_t = torch.cat(candidate_idxs_t, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps_t = overlaps[candidate_idxs_t, torch.arange(num_gt)]
        # t_overlaps = overlaps[candidate_idxs_t, torch.arange(num_gt)]

        t_diou = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps_t.mean(0)
        overlaps_std_per_gt = candidate_overlaps_t.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        #is_pos = (t_diou < overlaps_thr_per_gt[None, :]) & (t_diou >= 0.25 * overlaps_thr_per_gt[None, :])
        is_pos = t_diou >= 0.25 * overlaps_thr_per_gt[None, :]
        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side

        # is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]

        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)

        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        assigned_ld[
            max_overlaps != -INF] = max_overlaps[max_overlaps != -INF] + 0
        return assigned_ld

    def get_im_region1(
        self,
        bboxes,
        num_level_bboxes,
        gt_bboxes,
    ):
        """Assign gt to bboxes.
        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt
        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        self.topk = 9
        # compute iou between all bbox and gt

        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
        axmin = bboxes[:, 0] - bboxes[:, 2]/2
        aymin = bboxes[:, 1] - bboxes[:, 3]/2
        axmax = bboxes[:, 0] + bboxes[:, 2]/2
        aymax = bboxes[:, 1] + bboxes[:, 3]/2
        bboxes = torch.stack([axmin, aymin, axmax, aymax], dim=1)

        txmin = gt_bboxes[:, 0] - gt_bboxes[:, 2]/2
        tymin = gt_bboxes[:, 1] - gt_bboxes[:, 3]/2
        txmax = gt_bboxes[:, 0] + gt_bboxes[:, 2]/2
        tymax = gt_bboxes[:, 1] + gt_bboxes[:, 3]/2
        gt_bboxes = torch.stack([txmin, tymin, txmax, tymax], dim=1)

        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        #print('iou',overlaps)
        #diou = self.iou_calculator(bboxes, gt_bboxes, mode='diou')
        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assigned_fg = (assigned_gt_inds + 0).float()

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        candidate_idxs_t = []

        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_t = min(self.topk, bboxes_per_level)
            selectable_k = min(bboxes_per_level, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            _, topt_idxs_per_level = distances_per_level.topk(
                selectable_t, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            candidate_idxs_t.append(topt_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        candidate_idxs_t = torch.cat(candidate_idxs_t, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps_t = overlaps[candidate_idxs_t, torch.arange(num_gt)]
        # t_overlaps = overlaps[candidate_idxs_t, torch.arange(num_gt)]

        t_diou = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps_t.mean(0)
        overlaps_std_per_gt = candidate_overlaps_t.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        #is_pos = (t_diou < overlaps_thr_per_gt[None, :]) & (t_diou >= 0.25 * overlaps_thr_per_gt[None, :])
        is_pos = t_diou >= 0.5 * overlaps_thr_per_gt[None, :].max(0)[0]
        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side

        # is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]

        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)

        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        assigned_fg[
            max_overlaps != -INF] = max_overlaps[max_overlaps != -INF] + 0
        return assigned_fg

    def get_im_region(self, bboxes, gt_bboxes, mode='fitnet'):
        assert mode in ['gibox', 'finegrained', 'fitnet', 'decouple']
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
        axmin = bboxes[:, 0] - bboxes[:, 2]/2
        aymin = bboxes[:, 1] - bboxes[:, 3]/2
        axmax = bboxes[:, 0] + bboxes[:, 2]/2
        aymax = bboxes[:, 1] + bboxes[:, 3]/2
        bboxes = torch.stack([axmin, aymin, axmax, aymax], dim=1)

        txmin = gt_bboxes[:, 0] - gt_bboxes[:, 2]/2
        tymin = gt_bboxes[:, 1] - gt_bboxes[:, 3]/2
        txmax = gt_bboxes[:, 0] + gt_bboxes[:, 2]/2
        tymax = gt_bboxes[:, 1] + gt_bboxes[:, 3]/2
        gt_bboxes = torch.stack([txmin, tymin, txmax, tymax], dim=1)

        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        bboxes = bboxes[:, :4]
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assigned_fg = (assigned_gt_inds + 0).float()
        # compute iou between all bbox and gt


        iou = self.iou_calculator(bboxes, gt_bboxes, mode='iou')
        fine_grained = torch.nonzero(iou > 0.5 * iou.max(0)[0])
        assigned_fg[fine_grained[:, 0]] = 1

        '''
        gt_flag = torch.zeros(bboxes.shape[0])
        anchor_center = self.anchor_center(bboxes)
        for gt_bbox in gt_bboxes:
            in_gt_flag = torch.nonzero(
                (anchor_center[:, 0] > gt_bbox[0])
                & (anchor_center[:, 0] < gt_bbox[2])
                & (anchor_center[:, 1] > gt_bbox[1])
                & (anchor_center[:, 1] < gt_bbox[3]),
                as_tuple=False)
            gt_flag[in_gt_flag] = 1

        if mode == 'finegrained':
            return assigned_fg
        else:
            return gt_flag

        '''
        return assigned_fg