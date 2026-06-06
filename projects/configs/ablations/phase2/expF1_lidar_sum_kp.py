# ============================================================================
# Phase 2 F1.kp: LiDAR backbone + sum fusion + 多点采样(共享图像 key_points)
# ============================================================================
# Ablation row(单变量对比 F1):仅把 LiDAR BEV 采样从"anchor 中心 1 点"
# 改成"复用图像分支的 num_pts 个 key_points 各采 1 点 + mean pool"。
# 其他全部跟 F1 完全一致(sum 融合 + lidar_gate ReZero,无 MAFS、无 masked-modal)。
#
# 动机:
# - 物体中心在 BEV 上往往为空(LiDAR 反射多在轮廓/表面),单点采样命中率低
# - learnable key_points 会发散到物体边缘,多点 mean pool 能更稳定捕到回波
# - 与图像分支共享同一组 3D 采样位置,跨模态融合的几何对齐更一致
#
# 期望对比 F1_v2(干净 timestamp 数据下):
# - mAP / Recall 略升(LiDAR 信号更密实)
# - AMOTA 视情况(关联还是 IDS 主导,如 #2 paper-insights 说的)
# - 训练成本 ≈ F1(只多 13× grid_sample,O(ms) 量级,可忽略)
#
# Architecture diff vs F1:
#   - DFA.lidar_bev_sampling.multi_point_sampling = True
#   (零新增参数,只换采样几何)
# ============================================================================
_base_ = ["./expF1_lidar_sum.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            lidar_bev_sampling=dict(
                multi_point_sampling=True,
            ),
        ),
    ),
)
