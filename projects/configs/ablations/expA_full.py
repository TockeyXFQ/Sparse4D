# ============================================================================
# Phase 1 Stage 1b — A_full = Stage 1a ckpt + A1 Refine2D fine-tune
# ============================================================================
# Purpose
# -------
# 在 expA_full_no_refine2d.py (Stage 1a) 训出的 ckpt 上接 Refine2D fine-tune
# ~2000 iter,得到 4 个 trick 全开的最终 A_full ckpt。
#
# 训练食谱完全照搬 chenxi/onemodel 验证过的 "highlr" recipe(2026-05-15 实战
# 验证过,把 AMOTA 从 baseline 提到 0.7119):
#   1. 全局 lr 降 10x (4e-4 → 4e-5,backbone 几乎冻住)
#   2. aux_modules 单独 lr_mult=10 (Refine2D MLP 实际 lr=4e-4,from-scratch 级)
#   3. backbone / hybrid_encoder lr_mult=0.1 (实际 4e-6,冻得更死)
#   4. warmup_iters=100 (5% of 2000 iter,而非 baseline 的 25%)
#   5. min_lr_ratio=0.01 (短 fine-tune 不要 cosine tail 趋零)
#
# 不做这些就崩(chenxi 实战记录: from-scratch / 默认 schedule fine-tune 会让
# AMOTA 从 0.6818 跌到 0.4192,-0.26)。
#
# 启动命令
# -------
# # 0. 先确认 Stage 1a 的 ckpt 路径
# # 1. 启动 fine-tune (~3-4h on 8 GPU)
# USE_GLOBAL_PYTHON=1 \
#   CONFIG=projects/configs/ablations/expA_full.py \
#   WORK_DIR=work_dirs/expA_full \
#   bash scripts/03_train_baseline.sh
# ============================================================================
_base_ = ["./expA_full_no_refine2d.py"]

# ---- 加 Refine2D aux head ----
# Refine2D 在 chenxi 工程实证 gt_source='project' (geometric projection) 最稳定;
# 'real_simota' 模式会让长尾类崩盘,不要用。per-class loss weight 借用 chenxi
# 经验值(other_vehicle 3.0× / van 2.0× / 其他 1.0×),我们 R50 设定下不一定
# transferrable,Phase 1 完成后再做 7-point sweep 重新调。
model = dict(
    head=dict(
        aux_modules=dict(
            refine2d=dict(
                type="SparseBox2DRefinement",
                embed_dims=256,
                num_cls=10,
                box_mult_ratio=32.0,
                level_index=2,  # FPN stride-32 level
                with_cls_branch=True,
                gt_source="project",
                cls_prior_prob=0.01,
                loss_cls2d=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,
                ),
                loss_bbox2d=dict(type="IoULoss", loss_weight=2.0),
                # chenxi 经验值,可在 A_full ckpt 上做 7-point sweep 重调
                class_loss_weights=[
                    1.0,  # car
                    1.0,  # truck
                    3.0,  # construction_vehicle (chenxi 实证 boost 3x 最优)
                    1.0,  # bus
                    2.0,  # trailer (chenxi 实证 boost 2x 最优)
                    1.0,  # barrier
                    1.0,  # motorcycle
                    1.0,  # bicycle
                    1.0,  # pedestrian
                    1.0,  # traffic_cone
                ],
            ),
        ),
    ),
)

# ============================================================================
# Optimizer: full override(chenxi/onemodel highlr recipe, 字节对齐)
# ============================================================================
# 关键 trick: _delete_=True 强制完全替换父 config 的 optimizer dict。否则 mmcv
# 的 dict merge 会保留 baseline 的 paramwise_cfg.custom_keys (只有
# img_backbone:0.5),把我们的新 keys append 到末尾,而 mmcv 优化器构造器走
# 第一个匹配前缀,导致 head.aux_modules 被 head 这个更短前缀覆盖,lr_mult=10
# 完全失效。chenxi 工程踩过这个坑(参见 onemodel 同名 config 注释)。
optimizer = dict(
    _delete_=True,
    type="AdamW",
    # 全局 lr = baseline 的 1/10 (6e-4 → 6e-5)
    # backbone 实际 lr = 6e-5 * 0.1 = 6e-6,几乎冻住
    # aux_modules 实际 lr = 6e-5 * 10 = 6e-4 (= baseline from-scratch lr)
    lr=6e-5,
    weight_decay=0.001,
    paramwise_cfg=dict(
        # 顺序关键:aux_modules 前缀必须排在 head 前面,否则 head 匹配上把
        # aux_modules lr_mult 错误覆盖
        custom_keys={
            "head.aux_modules": dict(lr_mult=10.0),
            "img_backbone": dict(lr_mult=0.1),
            "img_neck": dict(lr_mult=1.0),
            "depth_branch": dict(lr_mult=1.0),
            "head": dict(lr_mult=1.0),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))

# Schedule: short fine-tune
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=100,  # 5% of 2000 iter
    warmup_ratio=1.0 / 3,
    min_lr_ratio=0.01,  # 不让 cosine tail 趋零
)

# 2000 iter ≈ 短 fine-tune(chenxi 工程实战值,经验上 1000-3000 iter 收敛)
runner = dict(type="IterBasedRunner", max_iters=2000)

# 高频 checkpoint 方便选最佳点
checkpoint_config = dict(interval=500, save_last=True, max_keep_ckpts=5)

# Load Stage 1a 的 ckpt(实际路径需要在跑 Stage 1a 后填,这里先用 placeholder)
# TODO: 跑完 Stage 1a 后改成实际路径,比如:
# load_from = "work_dirs/expA_full_no_refine2d/latest.pth"
load_from = "work_dirs/expA_full_no_refine2d/latest.pth"
