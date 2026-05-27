# ============================================================================
# Phase 1 leave-one-out A4 — Stage 1b (Refine2D fine-tune)
# ============================================================================
# Purpose: 从 expA_loo_a4_no_refine2d 的 ckpt 接 Refine2D fine-tune。
# Schedule + optimizer 跟 expA_full.py 一致。
# ============================================================================
_base_ = ["./expA_full.py"]

# 关掉 future reasoning(memory pool 仍然启用)
model = dict(
    head=dict(
        instance_bank=dict(future_reasoning_enable=False),
    ),
)

# 起点 ckpt 改成 loo_a4 的 Stage 1a
load_from = "work_dirs/expA_loo_a4_no_refine2d/latest.pth"
