# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        # ---- Phase 2 F1/F2: LiDAR BEV sampling + fusion ----
        # 默认 None = baseline 行为(camera-only,跟原版完全等价)
        # 启用时 dict(
        #     point_cloud_range=[-54, -54, -5, 54, 54, 3],   # BEV 物理范围
        #     lidar_channels=256,                            # BEV feature dim
        #     fusion_mode='sum'    # F1: features + lidar_features (P2 简单融合)
        #              | 'mafs'    # F2: MAFS attention 加权(自适应模态)
        # )
        lidar_bev_sampling: Optional[dict] = None,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

        # ============ Phase 2 F1/F2: LiDAR BEV sampling + fusion ============
        if lidar_bev_sampling is not None:
            self.lidar_bev_sampling = True
            self._lidar_pcr = lidar_bev_sampling.get(
                "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
            )
            lidar_channels = int(lidar_bev_sampling.get("lidar_channels", embed_dims))
            self._fusion_mode = lidar_bev_sampling.get("fusion_mode", "sum")
            assert self._fusion_mode in ("sum", "mafs"), (
                f"fusion_mode must be 'sum' (F1) or 'mafs' (F2), "
                f"got {self._fusion_mode}"
            )

            # 把 BEV feature channel 投到 embed_dims
            self.lidar_proj = Linear(lidar_channels, embed_dims)
            # LayerNorm 限制 LiDAR 特征幅度(关键!没有它,裸加的 lidar_features
            # 会无界增长,跨 6 个 decoder 层 + InstanceBank 时序 cache 累积,
            # 最终梯度爆炸 → NaN。BEVFusion/TransFusion/FUTR3D/CMT 都做归一化)
            self.lidar_norm = nn.LayerNorm(embed_dims)

            if self._fusion_mode == "mafs":
                # F2 MAFS: 每个 query 学 (w_img, w_lidar) 自适应权重
                # input = [instance_feature, image_features, lidar_features] (3*D)
                # output = (w_img, w_lidar) 经过 softmax。softmax 加权自带平衡,
                # 不需要 gate(否则 gate 在 mafs forward 不参与 → DDP unused param 崩)
                self.mafs_mlp = nn.Sequential(
                    Linear(3 * embed_dims, embed_dims),
                    nn.ReLU(inplace=True),
                    Linear(embed_dims, 2),
                )
                self.lidar_gate = None
            else:
                # 零初始化门控标量(ReZero / LayerScale):训练初期 LiDAR 贡献 = 0,
                # 模型逐渐学会用它,避免"从零初始化 + 高 lr"的新分支早期把预训练
                # image 路径带飞。仅 sum 模式创建(避免 mafs 模式 unused param)。
                self.lidar_gate = nn.Parameter(torch.zeros(1))
                self.mafs_mlp = None
        else:
            self.lidar_bev_sampling = False
            self.lidar_proj = None
            self.lidar_norm = None
            self.lidar_gate = None
            self.mafs_mlp = None

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features

        # ============ Phase 2 F1/F2: LiDAR BEV fusion ============
        # 在 output_proj 之前 fuse,这样 image / lidar features 在同一抽象层
        if self.lidar_bev_sampling and metas.get("lidar_bev") is not None:
            lidar_bev = metas["lidar_bev"]  # (B, C, H, W),来自 LiDAR backbone
            lidar_features = self._sample_lidar_bev(anchor, lidar_bev)
            # 防御:净化 inf/nan,堵住 LayerNorm 把单个 inf 放大成整个向量 nan
            # 的路径(已实测 LN(inf 输入)→ 全 nan)。正常数值不受影响。
            lidar_features = torch.nan_to_num(
                lidar_features, nan=0.0, posinf=0.0, neginf=0.0
            )
            # proj -> LayerNorm 归一化(限制幅度,防无界增长 → 梯度爆炸)
            lidar_features = self.lidar_norm(self.lidar_proj(lidar_features))
            # Cast 回 features 的 dtype(fp16 训练下 features 是 half;现 P2 用
            # fp32,此 cast 为 no-op,但保留以兼容未来 fp16)
            lidar_features = lidar_features.to(features.dtype)

            if self._fusion_mode == "sum":
                # F1: gated residual sum。lidar_gate 零初始化,训练初期 LiDAR
                # 贡献=0,逐渐 ramp up(ReZero),避免新分支早期带飞主网络
                features = features + self.lidar_gate.to(features.dtype) * lidar_features
            elif self._fusion_mode == "mafs":
                # F2 MAFS: 每个 query 学自适应权重 (image vs lidar)。softmax
                # 加权已自带平衡(w_img+w_lid=1),lidar_features 已 LayerNorm,
                # 无需额外 gate。
                w_input = torch.cat(
                    [instance_feature, features, lidar_features], dim=-1
                )
                w = self.mafs_mlp(w_input).softmax(dim=-1)  # (B, A, 2)
                features = (
                    w[..., 0:1] * features + w[..., 1:2] * lidar_features
                )

        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _sample_lidar_bev(self, anchor, lidar_bev):
        """每个 anchor 的 (x, y) 位置在 BEV feature map 上 bilinear sample.

        Args:
            anchor: (B, A, 11) — anchor 11 维 [x, y, z, w, l, h, sin, cos, vx, vy, vz]
            lidar_bev: (B, C, H, W) — LiDAR backbone 输出的 BEV feature

        Returns:
            sampled: (B, A, C) — 每个 anchor 在 BEV 上的采样 feature
        """
        # anchor xyz 在 LiDAR 坐标系。映射到 BEV grid 归一化坐标 [-1, 1]
        # 注意:LiDAR 点云的 (x, y) 经过 voxelize → SparseEncoder 后,
        # BEV feature map 的 W 维对应 x,H 维对应 y(或反过来,取决于实现)
        # mmdet3d 标准:H = (y_max - y_min) / voxel_y, W = (x_max - x_min) / voxel_x
        # grid_sample 期望 grid[..., 0] = W 方向 = x_norm,grid[..., 1] = H 方向 = y_norm
        pcr = self._lidar_pcr  # [x_min, y_min, z_min, x_max, y_max, z_max]
        x = anchor[..., 0]
        y = anchor[..., 1]
        x_norm = (x - pcr[0]) / (pcr[3] - pcr[0]) * 2.0 - 1.0
        y_norm = (y - pcr[1]) / (pcr[4] - pcr[1]) * 2.0 - 1.0
        # grid_sample input: lidar_bev (B, C, H, W); grid (B, A, 1, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)
        sampled = nn.functional.grid_sample(
            lidar_bev.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # sampled: (B, C, A, 1) → (B, A, C)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        return sampled

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
