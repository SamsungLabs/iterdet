import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .test_mixins import RPNTestMixin
from .single_stage import SumLayer


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector, RPNTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 iterative=False):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        self.iterative = iterative
        if self.iterative:
            self.history_transform = nn.Sequential(
                SumLayer(),
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            )

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img, history=None):
        """Directly extract features from the backbone+neck
        """
        if history is not None:
            history = self.history_transform(history)

        x = self.backbone(img, history)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img, history=None):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img, history)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      history=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img, history)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(
                *rpn_outs, img_metas, cfg=proposal_cfg)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if not self.iterative:
            x = self.extract_feat(img)

            if proposals is None:
                proposal_list = self.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals

            return self.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
        else:
            assert not self.with_mask
            height, width = img.shape[2:]
            history = torch.zeros((1, self.roi_head.bbox_head.num_classes - 1, height, width), device=img.device)
            det_bboxes = torch.zeros((0, 5), device=img.device)
            det_labels = torch.zeros((0,), dtype=torch.int64, device=img.device)
            for i in range(self.test_cfg.get('n_iterations', 1)):
                x = self.extract_feat(img, history)
                proposal_list = self.simple_test_rpn(x, img_metas)
                bboxes, labels = self.roi_head.simple_test_bboxes(
                    x, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale)
                if len(bboxes) == 0:
                    break
                det_bboxes = torch.cat((det_bboxes, bboxes), dim=0)
                det_labels = torch.cat((det_labels, labels), dim=0)
                for bbox, label in zip(bboxes, labels):
                    bbox = bbox * img_metas[0]['scale_factor']
                    x_min = torch.max(torch.round(bbox[0]).int(), torch.tensor(0, device=img.device).int())
                    y_min = torch.max(torch.round(bbox[1]).int(), torch.tensor(0, device=img.device).int())
                    x_max = torch.min(torch.round(bbox[2]).int(), torch.tensor(width, device=img.device).int() - 1)
                    y_max = torch.min(torch.round(bbox[3]).int(), torch.tensor(height, device=img.device).int() - 1)
                    history[0, label, y_min: y_max + 1, x_min: x_max + 1] += 1

            return bbox2result(det_bboxes, det_labels, self.roi_head.bbox_head.num_classes)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
