import torch
import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


class SumLayer(nn.Module):
    @staticmethod
    def forward(x):
        return torch.sum(x, dim=1, keepdim=True)


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 iterative=False):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.iterative = iterative
        if self.iterative:
            self.history_transform = nn.Sequential(
                SumLayer(),
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            )
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

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
        x = self.extract_feat(img, history)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      history=None):
        x = self.extract_feat(img, history)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def _single_simple_test(self, img, img_metas, rescale, history):
        x = self.extract_feat(img, history)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        return self.bbox_head.get_bboxes(*bbox_inputs)[0]

    def simple_test(self, img, img_metas, rescale=False):
        if not self.iterative:
            det_bboxes, det_labels = self._single_simple_test(img, img_metas, rescale, None)
        else:
            height, width = img.shape[2:]
            history = torch.zeros((1, self.bbox_head.num_classes - 1, height, width), device=img.device)
            det_bboxes = torch.zeros((0, 5), device=img.device)
            det_labels = torch.zeros((0,), dtype=torch.int64, device=img.device)
            for i in range(self.test_cfg.get('n_iterations', 1)):
                x = self.extract_feat(img, history)
                outs = self.bbox_head(x)
                bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
                bboxes, labels = self.bbox_head.get_bboxes(*bbox_inputs)[0]
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
        return bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
