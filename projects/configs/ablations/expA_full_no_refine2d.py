# ============================================================================
# Phase 1 Stage 1a — A_full with A2+A4+A5, without A1 Refine2D
# ============================================================================
# Purpose
# -------
# Phase 1 "全开" baseline(不含 A1 Refine2D):3 个 architectural tricks 一起
# from-scratch 训 100 epoch。Refine2D 是个 fine-tune trick(chenxi/onemodel 工程
# 实证 from-scratch 会崩),所以拆成两阶段:
#     Stage 1a (此 config) : 100 epoch from-scratch,含 A2 + A4 + A5
#     Stage 1b             : 从 Stage 1a ckpt fine-tune ~2000 iter,加 A1
#     ↓
#     最终 A_full = Stage 1b 的 ckpt(4 个 trick 全开)
#
# Tricks 配置详情
# --------------
# A1 Refine2D            : OFF (留到 Stage 1b)
# A2 MemoryBank          : ON  (num_memory_instances=1500 vs num_temp=600,
#                                2.5× 容量,memory_confidence_decay=0.85)
# A4 PF-Track Future     : ON  (PFTrackInstanceBank,hist_len=3, fut_len=1)
# A5 Birth/Death         : ON  (eval-only,推理时启用;此 config 用于训练,
#                                训练时 self.training=True 走默认 ID 分配)
#
# 启动命令
# -------
# USE_GLOBAL_PYTHON=1 CONFIG=projects/configs/ablations/expA_full_no_refine2d.py \
#   WORK_DIR=work_dirs/expA_full_no_refine2d \
#   bash scripts/03_train_baseline.sh
#
# Leave-one-out 关系
# -----------------
# loo A1 = 此 config 的 ckpt(免费,不需要单独训练)
# loo A2 = expA_loo_a2_no_refine2d.py (关 memory + 后续 Refine2D fine-tune)
# loo A4 = expA_loo_a4_no_refine2d.py (关 future + 后续 Refine2D fine-tune)
# loo A5 = expA_full 的 ckpt 评测时关 birth_death(免费)
# ============================================================================
_base_ = ["../sparse4dv3_temporal_r50_1x8_bs6_256x704.py"]

# ---- 替换 instance_bank: InstanceBank → PFTrackInstanceBank (A2+A4) ----
# PFTrackInstanceBank 继承 MemoryBank 继承 InstanceBank,所以同时支持
# num_memory_instances (A2) + future_reasoning_enable (A4) + birth_death (A5)。
model = dict(
    head=dict(
        instance_bank=dict(
            _delete_=True,  # 完全替换,不跟父 config 的 InstanceBank dict merge
            type="PFTrackInstanceBank",
            num_anchor=900,
            embed_dims=256,
            anchor="nuscenes_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600,
            confidence_decay=0.6,
            feat_grad=False,
            # ---- A2: long-horizon memory pool ----
            num_memory_instances=1500,  # 2.5× num_temp_instances
            memory_confidence_decay=0.85,
            # ---- A4: PF-Track Future Reasoning ----
            hist_len=3,
            fut_len=1,
            motion_mlp_hidden=256,
            future_reasoning_enable=True,
            # ---- A5: Birth/Death state machine (eval-only) ----
            birth_death=dict(
                min_hits=3,
                score_thr=0.4,
                max_age=5,
            ),
        ),
    ),
)
