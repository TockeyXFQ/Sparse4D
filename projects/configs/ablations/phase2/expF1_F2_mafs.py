# ============================================================================
# Phase 2 F1 + F2: LiDAR backbone + MAFS attention fusion
# ============================================================================
# Ablation row: F1 上 加 MAFS (Modality-Adaptive Feature Sampling),
# 把固定 sum fusion 换成 attention 加权(每个 query 自适应学 image vs lidar 权重)。
# 没有 F4 masked-modal training。
#
# Architecture diff vs F1:
#   - DFA.lidar_bev_sampling.fusion_mode='sum' -> 'mafs'
#   (新增一个 mafs_mlp,~130K params)
# ============================================================================
_base_ = ["./expF1_lidar_sum.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            lidar_bev_sampling=dict(
                fusion_mode="mafs",  # F2: 替换 F1 的 sum
            ),
        ),
    ),
)
