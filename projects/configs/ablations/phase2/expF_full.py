# ============================================================================
# Phase 2 F_full = F1 + F2 + F4 (Phase 2 最终 SOTA config)
# ============================================================================
# Ablation row: 完整 P2 多模态 SOTA。
#   F1 LiDAR backbone (sum fusion 已被 F2 升级成 MAFS)
#   F2 MAFS attention fusion
#   F4 Masked-modal training (训练时 30% 概率 zero-out image, 30% zero-out lidar,
#                              40% 双模态正常)
#
# Inference modes (同一 ckpt,paper 杀手锏):
#   - 默认 cam+lidar 双模态推理(论文 SOTA 数字)
#   - cam-only 推理(对标 DualViewDistill 0.669)→ 评测时人工 zero-out lidar
#   - lidar-only 推理(对标 CenterPoint ~0.69)→ 评测时人工 zero-out image
#
# Architecture diff vs F1+F2:
#   + masked_modal_prob=dict(image=0.3, lidar=0.3) 启用 F4
# ============================================================================
_base_ = ["./expF1_F2_mafs.py"]

model = dict(
    masked_modal_prob=dict(image=0.3, lidar=0.3),
)

# F4 masked-modal 必需:某个模态被随机 zero-out 的 step,该模态融合分支
# (lidar_proj / lidar_gate / mafs_mlp 等)不参与 loss → 无梯度。DDP 默认
# find_unused_parameters=False 会报 "Expected to have finished reduction ...
# parameters that were not used in producing loss"。开 True 让 DDP 每步动态
# 检测未用参数(性能略降 ~5-10%,但 masking 训练必须)。
# 注意:不能配 static_graph=True(masking 每步参与的参数集会变,与静态图冲突)。
find_unused_parameters = True
