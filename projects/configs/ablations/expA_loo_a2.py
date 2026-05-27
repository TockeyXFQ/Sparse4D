# ============================================================================
# Phase 1 leave-one-out A2 — Stage 1b (Refine2D fine-tune)
# ============================================================================
# Purpose: 从 expA_loo_a2_no_refine2d 的 ckpt 接 Refine2D fine-tune,得到
# "leave-out A2" 的最终评测点。
#
# Schedule + optimizer 跟 expA_full.py 一致(都是 chenxi/onemodel highlr recipe)。
# 唯一区别 = 起点 ckpt 不同。
# ============================================================================
_base_ = ["./expA_full.py"]

# 关掉 memory pool(从 expA_full 继承下来的 num_memory=1500 改回 600)
model = dict(
    head=dict(
        instance_bank=dict(num_memory_instances=600),
    ),
)

# 起点 ckpt 改成 loo_a2 的 Stage 1a
load_from = "work_dirs/expA_loo_a2_no_refine2d/latest.pth"
