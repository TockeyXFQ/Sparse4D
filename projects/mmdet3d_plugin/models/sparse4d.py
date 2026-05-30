# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature
from typing import Optional

import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

# Phase 2 F1 LiDAR backbone components - lazy import,只在配 LiDAR 时加载
try:
    from mmcv.ops import Voxelization
    VOXELIZATION_VALID = True
except ImportError:
    VOXELIZATION_VALID = False

__all__ = ["Sparse4D"]


@DETECTORS.register_module()
class Sparse4D(BaseDetector):
    """Sparse4D detector with optional Phase 2 LiDAR fusion.

    Phase 2 additions (跟 baseline 完全向后兼容,所有新参数默认 None):
        - pts_voxel_layer / pts_voxel_encoder / pts_middle_encoder /
          pts_backbone / pts_neck: LiDAR backbone 5 件套
          (HardSimpleVFE + SparseEncoder + SECOND + SECONDFPN,CenterPoint 同款)
        - masked_modal_prob: F4 masked-modal training 配置,
          dict(image=0.3, lidar=0.3) 表示训练时 30% 概率把 image feat zero-out,
          30% 概率把 lidar feat zero-out,40% 双模态正常

    当所有 P2 参数 = None 时,模型行为跟 baseline 完全一致。
    """

    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        # ---- Phase 2 F1: LiDAR backbone (可选,默认全 None = baseline 行为) ----
        pts_voxel_layer: Optional[dict] = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        # ---- Phase 2 F4: masked-modal training (可选) ----
        masked_modal_prob: Optional[dict] = None,
        # ---- Phase 2 调试:崩盘时打印 NaN 来源(image vs lidar 分支) ----
        # 默认 False(零开销);定位 NaN 时 config 里设 True,或启动命令加
        # --cfg-options model.debug_nan_source=True
        debug_nan_source: bool = False,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        self.debug_nan_source = debug_nan_source
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

        # ============ Phase 2 F1: LiDAR backbone build ============
        # 5 件套(CenterPoint 同款,标准 nuScenes setting):
        #   pts_voxel_layer    : Voxelization op (CUDA),把 (N, 5) 点云 → voxel
        #   pts_voxel_encoder  : HardSimpleVFE 把 voxel 内点 mean → voxel feature
        #   pts_middle_encoder : SparseEncoder (Sparse 3D Conv) → BEV feature
        #   pts_backbone       : SECOND 2D conv 处理 BEV → multi-scale BEV
        #   pts_neck           : SECONDFPN upsample → 单尺度 BEV (B, C, H, W)
        # 所有 5 个都 None 时,LiDAR 路径完全禁用,模型退化成 baseline。
        self._has_lidar_branch = pts_voxel_encoder is not None
        if self._has_lidar_branch:
            assert VOXELIZATION_VALID, (
                "mmcv.ops.Voxelization not available; cannot enable LiDAR branch"
            )
            assert all(x is not None for x in [
                pts_voxel_layer, pts_middle_encoder, pts_backbone, pts_neck
            ]), (
                "LiDAR branch requires all 5 of: pts_voxel_layer, "
                "pts_voxel_encoder, pts_middle_encoder, pts_backbone, pts_neck"
            )
            # SECOND / SECONDFPN 在 mmdet3d 的 registry 里(不在 mmdet),
            # 必须用 mmdet3d 的 build_backbone / build_neck
            from mmdet3d.models.builder import (
                build_backbone as build_pts_backbone,
                build_neck as build_pts_neck,
                build_voxel_encoder,
                build_middle_encoder,
            )
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
            self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)
            self.pts_backbone = build_pts_backbone(pts_backbone)
            self.pts_neck = build_pts_neck(pts_neck)

        # ============ Phase 2 F4: masked-modal training ============
        # dict(image=p_img, lidar=p_lid),训练时按概率 zero-out 模态。
        # 推理时整个机制是 no-op(eval 模式跳过,或者通过环境变量手动 mask)。
        self.masked_modal_prob = masked_modal_prob
        if masked_modal_prob is not None:
            assert "image" in masked_modal_prob or "lidar" in masked_modal_prob
            assert self._has_lidar_branch, (
                "masked_modal_prob requires LiDAR branch enabled"
            )

    # ============ Phase 2 F1: LiDAR feature extraction ============
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """List[(N_i, 5)] points -> (voxels, num_points_per_voxel, coors).

        Args:
            points: List of length B, 每个元素是 (N_i, 5) 的 LiDAR 点云
                    (x, y, z, intensity, sweep_index)

        Returns:
            voxels: (M, max_num_points, 5) — 所有 batch 的 voxel concat
            num_points: (M,) — 每个 voxel 实际点数
            coors: (M, 4) — [batch_idx, z, y, x] voxel 坐标
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_lidar_feat(self, points):
        """LiDAR points -> BEV feature.

        Args:
            points: List of length B, 每个元素是 (N_i, 5) 的累积 LiDAR 点云

        Returns:
            lidar_bev: (B, C, H, W) — BEV feature map,送给 head 用

        Note: LiDAR backbone (SparseEncoder + SECOND) 全程 fp32 运行,跟 fp16
        image backbone 混合训练。这是 mmdet3d / CenterPoint / BEVFusion 等 SOTA
        LiDAR 工作的标准做法 — spconv 的 fp16 支持不完整,fp32 最稳。
        Cast 到 fp32 后,LiDAR BEV feature 在 DFA `_sample_lidar_bev` 里再被
        cast 回 image features 的 dtype(默认 fp16)。
        """
        # 强制 fp32:即使 dataloader 在 fp16 hook 下传 fp16 points,这里也 cast
        points = [p.float() for p in points]
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # robust dtype align: 在每个 sub-module 之前,把 x cast 成该 module weight
        # 的 dtype。spconv (SparseEncoder) 内部可能输出 fp16(取决于版本),
        # 而 SECOND/SECONDFPN weight 不一定被 mmcv fp16 hook patch(因为它们注册
        # 在 mmdet3d registry 而非 mmdet registry,patch 范围不一定覆盖)。
        # 用 weight dtype 反推 input dtype 保证不论 fp32/fp16 都能跑。
        w_dtype = next(self.pts_backbone.parameters()).dtype
        x = x.to(w_dtype)
        x = self.pts_backbone(x)
        if self.pts_neck is not None:
            w_dtype = next(self.pts_neck.parameters()).dtype
            x = x.to(w_dtype) if isinstance(x, torch.Tensor) else [t.to(w_dtype) for t in x]
            x = self.pts_neck(x)
        # SECONDFPN 输出 list,取第 0 个(只有一层)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    # ============ Phase 2 F4: masked-modal training ============
    def _apply_masked_modal(self, feature_maps, lidar_bev):
        """训练时按概率 zero-out 某个模态的 feature。

        概率分配:
            p_image: 把 image feature_maps zero-out (only-LiDAR forward)
            p_lidar: 把 lidar_bev zero-out (only-Image forward)
            其余概率: 双模态正常 (sum to 1)
        """
        if not self.training or self.masked_modal_prob is None:
            return feature_maps, lidar_bev
        p_img = self.masked_modal_prob.get("image", 0.0)
        p_lid = self.masked_modal_prob.get("lidar", 0.0)
        rand = torch.rand(1).item()
        if rand < p_img:
            # zero-out image - 但保持 tensor shape / dtype / device 不变
            if isinstance(feature_maps, list):
                feature_maps = [torch.zeros_like(x) for x in feature_maps]
            else:
                feature_maps = torch.zeros_like(feature_maps)
        elif rand < p_img + p_lid:
            # zero-out lidar
            if lidar_bev is not None:
                lidar_bev = torch.zeros_like(lidar_bev)
        # else: 双模态正常
        return feature_maps, lidar_bev

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)

        # Phase 2 F1: LiDAR BEV feature extraction
        lidar_bev = None
        if self._has_lidar_branch and "points" in data:
            lidar_bev = self.extract_lidar_feat(data["points"])

        # Phase 2 调试:崩盘时定位 NaN 来源(image vs lidar 分支)。开销极小
        # (一次 isfinite reduce),只在 debug_nan_source=True 时启用。
        if self.debug_nan_source:
            self._check_finite("image_feature_maps", feature_maps)
            self._check_finite("lidar_bev", lidar_bev)

        # Phase 2 F4: masked-modal training (训练时按概率 zero-out 某模态)
        feature_maps, lidar_bev = self._apply_masked_modal(feature_maps, lidar_bev)

        # 喂给 head:lidar_bev 通过 data dict 传给 DFA(via head -> blocks.py)
        if lidar_bev is not None:
            data["lidar_bev"] = lidar_bev

        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    @staticmethod
    def _check_finite(name, x):
        """调试用:tensor / list[tensor] 出现 nan/inf 时打印来源 + 当前 sample。
        只在 model.debug_nan_source=True 时被调用。"""
        if x is None:
            return
        tensors = x if isinstance(x, (list, tuple)) else [x]
        for i, t in enumerate(tensors):
            if not torch.is_tensor(t):
                continue
            n_nan = torch.isnan(t).sum().item()
            n_inf = torch.isinf(t).sum().item()
            if n_nan or n_inf:
                print(
                    f"[NaN-DEBUG] {name}[{i}] has nan={n_nan} inf={n_inf} "
                    f"shape={tuple(t.shape)}",
                    flush=True,
                )

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        # Phase 2 F1: LiDAR BEV feature (eval 时也用,默认双模态推理)
        if self._has_lidar_branch and "points" in data:
            lidar_bev = self.extract_lidar_feat(data["points"])
            data["lidar_bev"] = lidar_bev

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
