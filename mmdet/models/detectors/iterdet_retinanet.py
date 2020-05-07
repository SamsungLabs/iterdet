import torch
from torch import nn

from mmdet.core import bbox2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class IterDetRetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(IterDetRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                               test_cfg, pretrained)
        self.history_transform = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
        )

    def extract_feat(self, img, history):
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
                      history=None):
        x = self.extract_feat(img, history)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        height, width = img.shape[2:]
        history = torch.zeros((1, 1, height, width), device=img.device)
        det_bboxes = torch.zeros((0, 5), device=img.device)
        det_labels = torch.zeros((0,), dtype=torch.int64, device=img.device)
        for i in range(self.test_cfg.n_iterations):
            x = self.extract_feat(img, history)
            outs = self.bbox_head(x)
            bboxes, labels = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)[0]
            if len(bboxes) == 0:
                break
            det_bboxes = torch.cat((det_bboxes, bboxes), dim=0)
            det_labels = torch.cat((det_labels, labels), dim=0)
            for bbox, label in zip(bboxes, labels):
                bbox = bbox * img_metas[0]['scale_factor'][0]  # TODO: Why 0 0?
                x_min = torch.max(torch.round(bbox[0]).int(), torch.tensor(0, device=img.device).int())
                y_min = torch.max(torch.round(bbox[1]).int(), torch.tensor(0, device=img.device).int())
                x_max = torch.min(torch.round(bbox[2]).int(), torch.tensor(width, device=img.device).int() - 1)
                y_max = torch.min(torch.round(bbox[3]).int(), torch.tensor(height, device=img.device).int() - 1)
                history[0, 0, y_min: y_max + 1, x_min: x_max + 1] += 1
        return bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
