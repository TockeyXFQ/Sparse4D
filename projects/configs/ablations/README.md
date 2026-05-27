# Phase 1 Ablation Configs — 用法说明

> Phase 1 实施 4 个 trick(A1 Refine2D / A2 MemoryBank / A4 PF-Track Future Reasoning / A5 Birth-Death),按 leave-one-out 出 ablation table。设计原则:**用尽量少的训练次数得到全部 5 行 ablation 数据**。

## Config 结构

```
projects/configs/ablations/
├── expA_full_no_refine2d.py        ← Stage 1a:A_full 训练起点(A2+A4+A5,from-scratch)
├── expA_full.py                    ← Stage 1b:A_full = Stage 1a ckpt + Refine2D fine-tune
├── expA_loo_a2_no_refine2d.py      ← Stage 1a 变体:关 A2 (memory)
├── expA_loo_a2.py                  ← Stage 1b 变体:关 A2 + Refine2D fine-tune
├── expA_loo_a4_no_refine2d.py      ← Stage 1a 变体:关 A4 (future)
├── expA_loo_a4.py                  ← Stage 1b 变体:关 A4 + Refine2D fine-tune
└── README.md                       ← 本文档
```

## 训练 + 评测顺序(总共 3 次 from-scratch + 3 次 fine-tune,~5 天)

### Step 1: 全开 A_full 训练(必做,最优先)

```bash
# 1a. From-scratch with A2+A4+A5 (~12-18h on 8×H20)
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/expA_full_no_refine2d.py \
  WORK_DIR=work_dirs/expA_full_no_refine2d \
  bash scripts/03_train_baseline.sh

# 1b. Refine2D fine-tune (~3-4h on 8×H20)
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/expA_full.py \
  WORK_DIR=work_dirs/expA_full \
  bash scripts/03_train_baseline.sh
```

### Step 2: Leave-one-out A2 (验证 MemoryBank 的边际贡献)

```bash
# 2a. From-scratch 关 A2
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/expA_loo_a2_no_refine2d.py \
  WORK_DIR=work_dirs/expA_loo_a2_no_refine2d \
  bash scripts/03_train_baseline.sh

# 2b. Refine2D fine-tune
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/expA_loo_a2.py \
  WORK_DIR=work_dirs/expA_loo_a2 \
  bash scripts/03_train_baseline.sh
```

### Step 3: Leave-one-out A4 (验证 PF-Track Future Reasoning 的边际贡献)

```bash
# 3a. From-scratch 关 A4
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/expA_loo_a4_no_refine2d.py \
  WORK_DIR=work_dirs/expA_loo_a4_no_refine2d \
  bash scripts/03_train_baseline.sh

# 3b. Refine2D fine-tune
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/expA_loo_a4.py \
  WORK_DIR=work_dirs/expA_loo_a4 \
  bash scripts/03_train_baseline.sh
```

## Leave-one-out 表(免费 / 需训练 / 评测点)

| 实验 | 配置 | 训练成本 | 评测命令 |
|---|---|---|---|
| **A0** baseline | 已完成 | — | `work_dirs/baseline_v3_r50_repro/latest.pth` |
| **A_full** | `expA_full.py` (Stage 1a + 1b) | from-scratch + fine-tune | `work_dirs/expA_full/latest.pth` |
| **loo A1 Refine2D** | `expA_full_no_refine2d.py` | **免费**(就是 Stage 1a 的 ckpt) | `work_dirs/expA_full_no_refine2d/latest.pth` |
| **loo A2 MemoryBank** | `expA_loo_a2.py` (Stage 1a + 1b) | from-scratch + fine-tune | `work_dirs/expA_loo_a2/latest.pth` |
| **loo A4 PF-Track** | `expA_loo_a4.py` (Stage 1a + 1b) | from-scratch + fine-tune | `work_dirs/expA_loo_a4/latest.pth` |
| **loo A5 Birth-Death** | A_full ckpt + eval 时 `birth_death=None` | **免费**(纯 post-process 改 flag) | 用 A_full ckpt,评测时改 config |

**实际训练次数:3 次 from-scratch + 3 次 fine-tune = 6 次训练 ≈ 5 天 8 卡满载**。  
loo A1 和 loo A5 不需要额外训练。

## 评测

```bash
# 评测 A_full
USE_GLOBAL_PYTHON=1 GPUS=8 \
  bash scripts/04_eval_my_baseline.sh work_dirs/expA_full/latest.pth

# 评测 loo A1 (= expA_full_no_refine2d 的 ckpt)
USE_GLOBAL_PYTHON=1 GPUS=8 \
  bash scripts/04_eval_my_baseline.sh work_dirs/expA_full_no_refine2d/latest.pth

# 评测 loo A5 (= A_full ckpt 但 eval-time 关 birth_death)
# 暂时需要手工改 config 里 instance_bank.birth_death=None 再跑评测,
# 或者在 04_eval_my_baseline.sh 加 --cfg-options override
```

## Phase 1 验收标准

按 EXPERIMENT_PLAN.html §3 表格,Phase 1 全开预期:

- **AMOTA**:0.55 ~ 0.58(vs baseline 0.442,涨 +0.10 ~ +0.14)
- **NDS**:0.59 ~ 0.61(vs baseline 0.559,涨 +0.03 ~ +0.05)
- **IDS**:< 300(vs baseline 499,降 ≥ 40%)

leave-one-out 后,每个 trick 的边际贡献应该是:
- A1 Refine2D:+0.02 ~ +0.03 AMOTA(per chenxi/onemodel 实战)
- A2 MemoryBank:+0.01 ~ +0.02 AMOTA(主要降长遮挡 IDS)
- A4 PF-Track Future:+0.02 ~ +0.03 AMOTA / IDS 降 ~30%
- A5 Birth-Death:IDS 进一步降 20-30%,AMOTA 略升
