# ============================================================================
# Phase 1 leave-one-out A2 (MemoryBank) — Stage 1a (no Refine2D yet)
# ============================================================================
# Purpose: 验证 MemoryBank 对最终数字的边际贡献。从 expA_full_no_refine2d 开始,
# 关掉 long-horizon memory pool(num_memory_instances 设为 = num_temp_instances,
# 此时 MemoryBank 退化成普通 ring-buffer 行为,等价于不开 A2)。
#
# 配套 fine-tune config: expA_loo_a2.py (会从此 config 的 ckpt fine-tune Refine2D)
# ============================================================================
_base_ = ["./expA_full_no_refine2d.py"]

model = dict(
    head=dict(
        instance_bank=dict(
            # num_memory_instances = num_temp_instances 退化成 ring-buffer
            # (MemoryBank.__init__ 里 max(num_memory, num_temp) 自动保护,但显式
            # 写出 600 让 config 一眼看明白这是 leave-out A2)
            num_memory_instances=600,
        ),
    ),
)
