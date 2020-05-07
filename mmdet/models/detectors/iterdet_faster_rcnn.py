import torch
from torch import nn

from mmdet.core import bbox2result
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class IterDetFasterRCNN(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(IterDetFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.history_transform = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
        )

    def extract_feat(self, img, history):
        """Directly extract features from the backbone+neck
        """
        history = self.history_transform(history)
        x = self.backbone(img, history)
        if self.with_neck:
            x = self.neck(x)
        return x

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

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        height, width = img.shape[2:]
        history = torch.zeros((1, 1, height, width), device=img.device)
        det_bboxes = torch.zeros((0, 5), device=img.device)
        det_labels = torch.zeros((0,), dtype=torch.int64, device=img.device)
        for i in range(self.test_cfg.n_iterations):
            x = self.extract_feat(img, history)
            proposal_list = self.simple_test_rpn(x, img_metas)
            bboxes, labels = self.roi_head.simple_test_bboxes(
                x, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale)
            if len(bboxes) == 0:
                break
            det_bboxes = torch.cat((det_bboxes, bboxes), dim=0)
            det_labels = torch.cat((det_labels, labels), dim=0)
            for bbox, label in zip(bboxes, labels):
                bbox = bbox * img_metas[0]['scale_factor'][0]  # TODO: Why 0 and 0?
                x_min = torch.max(torch.round(bbox[0]).int(), torch.tensor(0, device=img.device).int())
                y_min = torch.max(torch.round(bbox[1]).int(), torch.tensor(0, device=img.device).int())
                x_max = torch.min(torch.round(bbox[2]).int(), torch.tensor(width, device=img.device).int() - 1)
                y_max = torch.min(torch.round(bbox[3]).int(), torch.tensor(height, device=img.device).int() - 1)
                history[0, 0, y_min: y_max + 1, x_min: x_max + 1] += 1

        return bbox2result(det_bboxes, det_labels, self.roi_head.bbox_head.num_classes)
