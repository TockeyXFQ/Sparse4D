# ============================================================================
# Phase 1 leave-one-out A4 (PF-Track Future Reasoning) — Stage 1a (no Refine2D)
# ============================================================================
# Purpose: 验证 PF-Track Future Reasoning 对最终数字的边际贡献。从
# expA_full_no_refine2d 开始,关掉 future_reasoning_enable。注意此时
# PFTrackInstanceBank 仍然继承自 MemoryBank,memory pool 仍然启用(A2 仍生效),
# 只是不跑 motion_mlp 预测,等价于 MemoryBank 行为。
#
# 配套 fine-tune config: expA_loo_a4.py
# ============================================================================
_base_ = ["./expA_full_no_refine2d.py"]

model = dict(
    head=dict(
        instance_bank=dict(
            # 关掉 future reasoning,memory pool 保持启用 (num_memory=1500 不变)
            # 等价于直接用 MemoryBank
            future_reasoning_enable=False,
        ),
    ),
)
