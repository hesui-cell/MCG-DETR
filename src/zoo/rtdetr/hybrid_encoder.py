'''RT-DETR Hybrid Encoder with CSGMSF and MCD-Adapter.

MCD-Adapter: Material-Contour Decoupling Adapter for occluded X-ray imagery.
CSGMSF: Channel-Spatial Gated Multi-Scale Fusion.
'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from ...core import register


__all__ = ['HybridEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None,
                 bias=False, act=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            groups=groups,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SpatialAttention(nn.Module):
    """Lightweight spatial gate used by CSGMSF."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CSGMSF(nn.Module):
    """Channel-Spatial Gated Multi-Scale Fusion.

    The module replaces a fixed concatenation at FPN/PAN junctions with:
      1) pooled multi-scale context for channel-wise source selection;
      2) a spatial gate for suppressing irrelevant/occluded responses.

    X and Y must have the same shape [B, C, H, W].
    """
    def __init__(self, in_channels, pool_sizes=(1, 2, 4), dropout=0.1,
                 shuffle_groups=4):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(s) for s in self.pool_sizes
        ])

        # Concatenated X/Y has 2C channels. Each pooled branch keeps 2C.
        concat_channels = 2 * in_channels * len(self.pool_sizes)
        hidden = max(in_channels // 4, 16)
        self.context_reduce = nn.Conv2d(concat_channels, hidden, 1, bias=False)
        self.context_norm = nn.BatchNorm2d(hidden)
        self.context_act = nn.SiLU(inplace=True)

        self.shuffle_groups = shuffle_groups if hidden % shuffle_groups == 0 else 1
        self.dropout = nn.Dropout2d(dropout)
        self.weight_conv = nn.Conv2d(hidden, in_channels, 1, bias=True)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    @staticmethod
    def _channel_shuffle(x, groups):
        if groups == 1:
            return x
        n, c, h, w = x.shape
        x = x.reshape(n, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n, c, h, w)

    def forward(self, x, y):
        if x.shape != y.shape:
            raise ValueError(
                f'CSGMSF expects equal feature shapes, got {x.shape} and {y.shape}.')

        cat = torch.cat([x, y], dim=1)
        pooled = [
            F.interpolate(pool(cat), size=cat.shape[-2:], mode='nearest')
            for pool in self.pools
        ]
        context = torch.cat(pooled, dim=1)
        context = self.context_act(self.context_norm(self.context_reduce(context)))
        context = self._channel_shuffle(context, self.shuffle_groups)

        sigma_c = torch.sigmoid(self.weight_conv(self.dropout(context)))
        fused = 2.0 * sigma_c * x + 2.0 * (1.0 - sigma_c) * y

        sigma_s = self.spatial_attention(fused)
        return fused * sigma_s


class MCDAdapter(nn.Module):
    """Material-Contour Decoupling Adapter (MCD-Adapter).

    Motivation for OPIXray:
      * object shape/contours are often preserved under X-ray projection;
      * material/color responses overlap with clutter and can couple with edges;
      * processing every channel with a large spatial operator is unnecessary.

    Design:
      1) process only a fraction of channels (partial-channel design);
      2) split features into low-frequency material context and high-frequency detail;
      3) orthogonalize detail against material context to reduce feature coupling;
      4) model elongated knife contours using horizontal/vertical strip depthwise convs;
      5) re-inject the update through a small LayerScale-style residual coefficient.
    """
    def __init__(self, channels, partial_ratio=0.25, strip_kernel=7,
                 max_gain=0.20, eps=1e-6):
        super().__init__()
        if not 0.0 < partial_ratio <= 1.0:
            raise ValueError('partial_ratio must be in (0, 1].')
        if strip_kernel % 2 == 0:
            raise ValueError('strip_kernel must be odd.')

        partial_channels = max(int(channels * partial_ratio), 16)
        partial_channels = min(partial_channels, channels)
        self.partial_channels = partial_channels
        self.max_gain = float(max_gain)
        self.eps = eps

        p = partial_channels
        pad = strip_kernel // 2

        # Direction-aware contour modeling on the decoupled high-frequency branch.
        self.horizontal = nn.Conv2d(
            p, p, kernel_size=(1, strip_kernel), padding=(0, pad),
            groups=p, bias=False)
        self.vertical = nn.Conv2d(
            p, p, kernel_size=(strip_kernel, 1), padding=(pad, 0),
            groups=p, bias=False)
        self.diagonal_local = nn.Conv2d(
            p, p, kernel_size=3, padding=1, groups=p, bias=False)
        self.contour_norm = nn.BatchNorm2d(p)

        # Low-frequency material/context branch.
        self.material_refine = nn.Sequential(
            nn.Conv2d(p, p, kernel_size=3, padding=1, groups=p, bias=False),
            nn.BatchNorm2d(p),
            nn.SiLU(inplace=True),
        )
        squeeze = max(p // 4, 8)
        self.material_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(p, squeeze, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze, p, 1, bias=True),
            nn.Sigmoid(),
        )

        # A one-channel confidence map selects contour vs. material evidence by location.
        self.spatial_selector = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # Expand the partial-channel update to all encoder channels.
        self.update_proj = nn.Sequential(
            nn.Conv2d(p, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

        # Small but non-zero LayerScale coefficient: nearly identity at initialization,
        # while still allowing all branch parameters to receive gradients immediately.
        self.alpha = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

    def _orthogonalize(self, detail, material):
        # Per-sample, per-channel projection of detail onto material.
        numerator = (detail * material).mean(dim=(2, 3), keepdim=True)
        denominator = material.square().mean(dim=(2, 3), keepdim=True) + self.eps
        return detail - (numerator / denominator) * material

    def forward(self, x):
        p = self.partial_channels
        x_partial = x[:, :p, :, :]

        # Local low/high-frequency decomposition without changing resolution.
        material = F.avg_pool2d(x_partial, kernel_size=3, stride=1, padding=1)
        detail = x_partial - material
        detail = self._orthogonalize(detail, material)

        contour = (
            self.horizontal(detail)
            + self.vertical(detail)
            + self.diagonal_local(detail)
        )
        contour = self.act(self.contour_norm(contour))

        material = self.material_refine(material)
        material = material * self.material_gate(material)

        contour_cue = contour.abs().mean(dim=1, keepdim=True)
        material_cue = material.abs().mean(dim=1, keepdim=True)
        selector = self.spatial_selector(torch.cat([contour_cue, material_cue], dim=1))

        fused_partial = selector * contour + (1.0 - selector) * material
        update = self.update_proj(fused_partial)

        gain = self.max_gain * torch.tanh(self.alpha)
        return x + gain * update


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3,
                 expansion=1.0, bias=None, act='silu'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1,
                                   bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1,
                                   bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        self.conv3 = (
            ConvNormLayer(hidden_channels, out_channels, 1, 1,
                          bias=bias, act=act)
            if hidden_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        x_1 = self.bottlenecks(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


@register()
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_csgmsf=True,
                 use_mcd_adapter=True,
                 mcd_levels=[0],
                 mcd_partial_ratio=0.25,
                 mcd_strip_kernels=[7],
                 mcd_max_gains=[0.20]):

        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.use_csgmsf = use_csgmsf
        self.use_mcd_adapter = use_mcd_adapter

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # Standard channel projection retained for compatibility with pretrained RT-DETR.
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim)
            )
            for in_channel in in_channels
        ])

        # MCD-Adapter is only placed on high-resolution P3/P4 by default.
        self.mcd_adapters = nn.ModuleList([nn.Identity() for _ in in_channels])
        if use_mcd_adapter:
            if len(mcd_strip_kernels) != len(mcd_levels):
                raise ValueError('mcd_strip_kernels length must match mcd_levels.')
            if len(mcd_max_gains) != len(mcd_levels):
                raise ValueError('mcd_max_gains length must match mcd_levels.')
            for level, kernel, gain in zip(mcd_levels, mcd_strip_kernels, mcd_max_gains):
                if level < 0 or level >= len(in_channels):
                    raise ValueError(f'Invalid MCD level index: {level}.')
                self.mcd_adapters[level] = MCDAdapter(
                    hidden_dim,
                    partial_ratio=mcd_partial_ratio,
                    strip_kernel=kernel,
                    max_gain=gain)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # Top-down FPN.
        self.lateral_convs = nn.ModuleList()
        self.fpn_fusions = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_fusions.append(
                CSGMSF(hidden_dim) if use_csgmsf else nn.Identity())
            fpn_in = hidden_dim if use_csgmsf else hidden_dim * 2
            self.fpn_blocks.append(
                CSPRepLayer(fpn_in, hidden_dim, round(3 * depth_mult),
                            act=act, expansion=expansion))

        # Bottom-up PAN.
        self.downsample_convs = nn.ModuleList()
        self.pan_fusions = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_fusions.append(
                CSGMSF(hidden_dim) if use_csgmsf else nn.Identity())
            pan_in = hidden_dim if use_csgmsf else hidden_dim * 2
            self.pan_blocks.append(
                CSPRepLayer(pan_in, hidden_dim, round(3 * depth_mult),
                            act=act, expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256,
                                            temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

    def _fuse(self, fusion_module, x, y):
        if self.use_csgmsf:
            return fusion_module(x, y)
        return torch.cat([x, y], dim=1)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        proj_feats = [
            self.input_proj[i](feat) for i, feat in enumerate(feats)
        ]
        if self.use_mcd_adapter:
            proj_feats = [
                self.mcd_adapters[i](feat) for i, feat in enumerate(proj_feats)
            ]

        # AIFI / intra-scale interaction.
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(
                        src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(
                    -1, self.hidden_dim, h, w).contiguous()

        # Top-down FPN.
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            block_idx = len(self.in_channels) - 1 - idx
            feat_high = self.lateral_convs[block_idx](inner_outs[0])
            inner_outs[0] = feat_high
            feat_low = proj_feats[idx - 1]
            upsample_feat = F.interpolate(
                feat_high, size=feat_low.shape[-2:], mode='nearest')
            fused = self._fuse(self.fpn_fusions[block_idx], upsample_feat, feat_low)
            inner_out = self.fpn_blocks[block_idx](fused)
            inner_outs.insert(0, inner_out)

        # Bottom-up PAN.
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if downsample_feat.shape[-2:] != feat_high.shape[-2:]:
                downsample_feat = F.interpolate(
                    downsample_feat, size=feat_high.shape[-2:], mode='nearest')
            fused = self._fuse(self.pan_fusions[idx], downsample_feat, feat_high)
            out = self.pan_blocks[idx](fused)
            outs.append(out)

        return outs