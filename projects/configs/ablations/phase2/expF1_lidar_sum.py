# ============================================================================
# Phase 2 F1: LiDAR backbone + 双模态 sampling (sum fusion)
# ============================================================================
# Ablation row: baseline + LiDAR backbone (CenterPoint 同款) + 简单 sum fusion
# 没有 F2 MAFS,没有 F4 masked-modal training。
#
# Architecture changes vs baseline:
#   + pts_voxel_layer + pts_voxel_encoder + pts_middle_encoder + pts_backbone + pts_neck
#       (CenterPoint nuScenes config 同款 5 件套, 输出 BEV feature 256ch)
#   + DFA.lidar_bev_sampling=dict(fusion_mode='sum') 启用 BEV sampling + sum fusion
#   + train/test_pipeline 加 LoadPointsFromMultiSweepsSparse4D (10 sweeps 累积)
#   + Collect.keys 加 'points'
#
# 启动命令:
#   USE_GLOBAL_PYTHON=1 \
#     CONFIG=projects/configs/ablations/phase2/expF1_lidar_sum.py \
#     WORK_DIR=work_dirs/expF1_lidar_sum \
#     bash scripts/03_train_baseline.sh
# ============================================================================
_base_ = ["../../sparse4dv3_temporal_r50_1x8_bs6_256x704.py"]

# ============ LiDAR 配置(全 P2 ablation 共享)============
# nuScenes 标准 setting:point_cloud_range = [-54, -54, -5, 54, 54, 3]
# voxel_size = [0.075, 0.075, 0.2] → BEV grid = 1440 × 1440 × 41
# (跟 CenterPoint / TransFusion / BEVFusion 同 setting,数字可比)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
sparse_shape = [
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]) + 1,  # 41
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),       # 1440
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),       # 1440
]
lidar_bev_channels = 256  # SECONDFPN 输出的 BEV channel 数 (256 来自 [128, 256] sum)

# ============ Model ============
model = dict(
    # ---- LiDAR backbone (CenterPoint 同款 5 件套) ----
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
    ),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=5),
    pts_middle_encoder=dict(
        type="SparseEncoder",
        in_channels=5,
        sparse_shape=sparse_shape,
        output_channels=128,
        order=("conv", "norm", "act"),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type="basicblock",
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        out_channels=[128, 128],
        upsample_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    # F1 不开 masked_modal_prob(此处不写,model 默认 None;
    # F_full config 在 child 里加 dict 才不会触发 mmcv None→dict 类型冲突)
    # ---- DFA 加 LiDAR BEV sampling (F1 sum fusion) ----
    head=dict(
        deformable_model=dict(
            lidar_bev_sampling=dict(
                point_cloud_range=point_cloud_range,
                lidar_channels=lidar_bev_channels,
                fusion_mode="sum",  # F1
            ),
        ),
    ),
)

# ============ Dataset Pipeline ============
# 关键改动:
#   1. LoadPointsFromMultiSweepsSparse4D 累积 10 sweep
#   2. Collect.keys 加 'points'
file_client_args = dict(backend="disk")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
strides = [4, 8, 16, 32]
num_depth_layers = 3
class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5, use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweepsSparse4D",
        sweeps_num=10, load_dim=5, use_dim=[0, 1, 2, 3, 4], time_dim=4,
        pad_empty_sweeps=True, remove_close=True, test_mode=False,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "points",  # P2 新增
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5, use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweepsSparse4D",
        sweeps_num=10, load_dim=5, use_dim=[0, 1, 2, 3, 4], time_dim=4,
        pad_empty_sweeps=True, remove_close=True, test_mode=True,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "points",  # P2 新增
            "timestamp",
            "projection_mat",
            "image_wh",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

# 用 input_modality 标识 use_lidar (会影响 dataset 类的 modality 检查)
input_modality = dict(
    use_lidar=True,  # P2 关键改动:启用 LiDAR
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data = dict(
    train=dict(
        modality=input_modality,
        pipeline=train_pipeline,
    ),
    val=dict(
        modality=input_modality,
        pipeline=test_pipeline,
    ),
    test=dict(
        modality=input_modality,
        pipeline=test_pipeline,
    ),
)

# ============ 全 fp32 训练(关掉 fp16 混合精度)============
# 原因:baseline 用静态 fp16=dict(loss_scale=32.0)。静态 loss scale 在梯度出现
# inf/nan 时**不会跳过 optimizer step**,直接 step → 权重永久 NaN 污染。
# baseline 纯 image 路径在 loss_scale=32 下数值稳定,但 P2 新增的 LiDAR 融合
# 路径(grid_sample → spconv SparseEncoder,fp16 激活)backward 时偶发溢出,
# 触发上述污染 → 实测 2 机 16 卡训练从 iter 51 起 loss 全程 NaN。
#
# 全 fp32 确定性消除所有 fp16 溢出可能(fp32 前向已验证 clean)。代价:显存约
# +50%、速度约 1.5-2× 慢,H20 (97GB) 完全够。
#
# 覆盖 base 的 fp16=dict(loss_scale=32.0) → None。mmcv 允许 child dict→None
# (注意跟 lessons #2 的 None→dict 方向相反)。fp16=None 时 runner 用普通
# OptimizerHook,代码里的 @auto_fp16/@force_fp32 装饰器检测 fp16_enabled=False
# 自动 no-op(纯 fp32 运行),无副作用。
#
# 【速度优先的替代方案】如果以后想恢复 fp16 加速,不要用静态 loss_scale,
# 改成动态:fp16 = dict(loss_scale='dynamic')。动态 scale 检测到 inf/nan 梯度
# 会自动跳过该 step + 降 scale,避免权重污染(但需重新验证 LiDAR 路径稳定性)。
fp16 = None

# ============ NaN 自动早停 ============
# 一旦 loss 变 nan/inf 立即终止训练(P2 调试期多次因 NaN 没及时发现白烧多机
# GPU)。patience=0 = 第一次 nan 就停。配合 model.debug_nan_source=True 可在
# 崩盘瞬间定位是 image 还是 lidar 分支先坏。
custom_hooks = [
    dict(type="NaNStopHook", check_loss=True, check_grad=True, patience=0),
]

# Note: LiDAR backbone 显存比 image-only baseline 重 (+SparseEncoder ~7M params
# + 大 BEV feature map),如果 OOM 把 samples_per_gpu 从 6 降到 4
# data = dict(samples_per_gpu=4)
