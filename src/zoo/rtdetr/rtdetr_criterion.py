"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Ultimate RT-DETR Criterion for X-ray Prohibited Item Detection.
Integrated with:
1. ASB-Lite: Adaptive Scale-Boundary Loss (for elongated/thin items).
2. WIoU (Wise-IoU): Replaces GIoU to handle heavily occluded/overlapping items.
3. NWD (Normalized Wasserstein Distance): For micro/tiny objects (razor blades, pills).
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


# ============================================================
# 🌟 专门针对 X 光特征的新增 Loss 算子
# ============================================================
def wiou_loss(pred_boxes, target_boxes, pred_xyxy, target_xyxy):
    """
    Wise-IoU (WIoU) v1.
    极其适合 X 光图像中边界模糊、严重遮挡的困难样本。
    """
    ious, _ = box_iou(pred_xyxy, target_xyxy)
    iou = torch.diag(ious)

    p_cx, p_cy = pred_boxes[:, 0], pred_boxes[:, 1]
    t_cx, t_cy = target_boxes[:, 0], target_boxes[:, 1]

    # 外接框 (Enclosing box)
    lt = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=1e-6)

    # 中心点距离惩罚
    dist2 = (p_cx - t_cx)**2 + (p_cy - t_cy)**2
    enclose2 = wh[:, 0]**2 + wh[:, 1]**2

    # WIoU 核心距离放大系数
    r_wiou = torch.exp(dist2 / enclose2.clamp(min=1e-6))

    return r_wiou * (1.0 - iou)


def nwd_loss(pred_boxes, target_boxes, constant=12.8):
    """
    Normalized Wasserstein Distance (NWD) Loss.
    将边界框建模为 2D 高斯分布。解决微小物体（如刀片）IoU 骤降为 0 的梯度断裂问题。
    """
    p_cx, p_cy, p_w, p_h = pred_boxes.unbind(-1)
    t_cx, t_cy, t_w, t_h = target_boxes.unbind(-1)

    # Wasserstein Distance 针对边界框的近似计算 (W2^2)
    center_dist = (p_cx - t_cx)**2 + (p_cy - t_cy)**2
    wh_dist = ((p_w - t_w)**2 + (p_h - t_h)**2) / 4.0
    w2 = center_dist + wh_dist

    # 归一化 (Normalized)
    nwd = torch.exp(-torch.sqrt(w2 + 1e-6) / constant)

    return 1.0 - nwd


# ============================================================
# 原生及增强版 RTDETRCriterion
# ============================================================
@register()
class RTDETRCriterion(nn.Module):
    """Compute Hungarian-matched classification and localization losses."""

    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        eos_coef=1e-4,
        num_classes=80,
        # ASB-Loss hyperparameters
        asb_shape_weight=0.50,
        asb_hard_gain=0.50,
        asb_scale_gain=0.25,
        asb_focusing_gamma=2.0,
        asb_boundary_beta=0.10,
        asb_shape_beta=0.20,
        asb_max_normalized_error=4.0,
        asb_eps=1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma

        # ASB Init
        self.asb_shape_weight = float(asb_shape_weight)
        self.asb_hard_gain = float(asb_hard_gain)
        self.asb_scale_gain = float(asb_scale_gain)
        self.asb_focusing_gamma = float(asb_focusing_gamma)
        self.asb_boundary_beta = float(asb_boundary_beta)
        self.asb_shape_beta = float(asb_shape_beta)
        self.asb_max_normalized_error = float(asb_max_normalized_error)
        self.asb_eps = float(asb_eps)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t['labels'][j] for t, (_, j) in zip(targets, indices)
        ])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(
                src_logits[idx], target_classes_o
            )[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t['labels'][j] for t, (_, j) in zip(targets, indices)
        ])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1
        )[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits,
            target,
            self.alpha,
            self.gamma,
            reduction='none',
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        if src_boxes.numel() == 0:
            ious = src_boxes.new_zeros((0,))
        else:
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
            ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([
            t['labels'][j] for t, (_, j) in zip(targets, indices)
        ])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1
        )[..., :-1]

        target_score_o = torch.zeros_like(
            target_classes, dtype=src_logits.dtype
        )
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = torch.sigmoid(src_logits).detach()
        weight = (
            self.alpha * pred_score.pow(self.gamma) * (1 - target)
            + target_score
        )
        loss = F.binary_cross_entropy_with_logits(
            src_logits,
            target_score,
            weight=weight,
            reduction='none',
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v['labels']) for v in targets], device=device
        )
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    # ============================================================
    # 🌟 核心升级区域：边界框回归 Loss 改造
    # ============================================================
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Modified Box Loss for X-ray:
        - Keeps basic L1 Loss.
        - Replaces GIoU with WIoU (better for severe occlusion).
        - Adds NWD Loss (better for micro items).
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        if src_boxes.numel() == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            # Return keys must match weight_dict in your yaml config
            return {'loss_bbox': zero, 'loss_giou': zero}

        # 1. 基础 L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)

        # 2. 🌟 WIoU: 替代原有的 GIoU，应对严重遮挡和边界模糊
        # 注意：这里我们依然使用 'loss_giou' 作为字典 Key，
        # 这样你就不需要去修改 YAML 配置文件中的 weight_dict 键名了，它会自动生效。
        loss_wiou = wiou_loss(src_boxes, target_boxes, src_xyxy, tgt_xyxy)

        # 3. 🌟 NWD Loss (可选): 专门辅助回归极小目标
        # loss_nwd = nwd_loss(src_boxes, target_boxes)

        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_wiou.sum() / num_boxes,
            # 如果你在 yaml 中配置了 weight_dict={'loss_nwd': 1.0}，可以解除下面这行的注释
            # 'loss_nwd': loss_nwd.sum() / num_boxes,
        }

    def loss_asb(self, outputs, targets, indices, num_boxes):
        """Ultra-light Adaptive Scale-Boundary Loss (ASB-Lite)."""
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        if src_boxes.numel() == 0:
            return {'loss_asb': outputs['pred_boxes'].sum() * 0.0}

        eps = self.asb_eps
        max_err = self.asb_max_normalized_error

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)

        tgt_wh = target_boxes[:, 2:].clamp_min(eps)
        tgt_w = tgt_wh[:, 0]
        tgt_h = tgt_wh[:, 1]

        # Normalize horizontal edges by GT width and vertical edges by GT height.
        edge_normalizer = torch.stack(
            [tgt_w, tgt_h, tgt_w, tgt_h],
            dim=-1,
        )

        normalized_edge_error = (
            (src_xyxy - tgt_xyxy) / edge_normalizer
        ).clamp(min=-max_err, max=max_err)

        boundary_loss = F.smooth_l1_loss(
            normalized_edge_error,
            torch.zeros_like(normalized_edge_error),
            beta=self.asb_boundary_beta,
            reduction='none',
        ).mean(dim=-1)

        # Cheap GT-only weights. No predicted-box IoU is computed.
        with torch.no_grad():
            area = (tgt_w * tgt_h).clamp(0.0, 1.0)
            scale_weight = 1.0 + self.asb_scale_gain * (
                1.0 - torch.sqrt(area + eps)
            )

            short_side = torch.minimum(tgt_w, tgt_h)
            long_side = torch.maximum(tgt_w, tgt_h)
            elongation = 1.0 - short_side / long_side.clamp_min(eps)
            shape_weight = 1.0 + self.asb_shape_weight * elongation

            adaptive_weight = scale_weight * shape_weight
            adaptive_weight = adaptive_weight / adaptive_weight.mean().clamp_min(eps)

        loss_asb = (adaptive_weight * boundary_loss).sum() / num_boxes
        return {'loss_asb': loss_asb}

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'asb': self.loss_asb,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](
            outputs, targets, indices, num_boxes, **kwargs
        )

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if 'aux' not in k
        }

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(
            num_boxes / get_world_size(), min=1
        ).item()

        indices = self.matcher(outputs_without_aux, targets)['indices']

        losses = {}
        for loss in self.losses:
            loss_kwargs = {'log': True} if loss == 'labels' else {}
            l_dict = self.get_loss(
                loss, outputs, targets, indices, num_boxes, **loss_kwargs
            )
            l_dict = {
                k: value * self.weight_dict[k]
                for k, value in l_dict.items()
                if k in self.weight_dict
            }
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)['indices']
                for loss in self.losses:
                    if loss in ('masks', 'asb'):
                        continue
                    loss_kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_boxes,
                        **loss_kwargs,
                    )
                    l_dict = {
                        k: value * self.weight_dict[k]
                        for k, value in l_dict.items()
                        if k in self.weight_dict
                    }
                    l_dict = {
                        k + f'_aux_{i}': value for k, value in l_dict.items()
                    }
                    losses.update(l_dict)

        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs
            indices = self.get_cdn_matched_indices(
                outputs['dn_meta'], targets
            )
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss in ('masks', 'asb'):
                        continue
                    loss_kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        dn_num_boxes,
                        **loss_kwargs,
                    )
                    l_dict = {
                        k: value * self.weight_dict[k]
                        for k, value in l_dict.items()
                        if k in self.weight_dict
                    }
                    l_dict = {
                        k + f'_dn_{i}': value for k, value in l_dict.items()
                    }
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx = dn_meta['dn_positive_idx']
        dn_num_group = dn_meta['dn_num_group']
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(
                    num_gt, dtype=torch.int64, device=device
                ).tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                empty = torch.zeros(
                    0, dtype=torch.int64, device=device
                )
                dn_match_indices.append((empty, empty))
        return dn_match_indices


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res