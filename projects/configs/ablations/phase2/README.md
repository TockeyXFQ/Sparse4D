# Phase 2 Ablation Configs — 多模态融合(LiDAR + Camera)

## Phase 2 论文 contribution

**"Unified Sparse Tracking across Camera, LiDAR, and Mixed Modalities"** — 一个 ckpt 同时支持:
- **Cam+LiDAR 推理**(SOTA 主战场)
- **Cam-only 推理**(对标 DualViewDistill 0.669)
- **LiDAR-only 推理**(对标 CenterPoint ~0.69)

## Config 结构

```
projects/configs/ablations/phase2/
├── expF1_lidar_sum.py        ← F1: 加 LiDAR backbone, simple sum fusion
├── expF1_F2_mafs.py          ← F1+F2: MAFS attention fusion (替换 sum)
└── expF_full.py              ← F1+F2+F4: 加 masked-modal training (论文 SOTA)
```

config 是 cascade 继承的:`expF_full.py → expF1_F2_mafs.py → expF1_lidar_sum.py → baseline`,
每个新 config 只加一个 trick,leave-one-out 容易。

## 训练 + 评测顺序

### Step 1:F1 (基线 + LiDAR sum fusion)

```bash
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/phase2/expF1_lidar_sum.py \
  WORK_DIR=work_dirs/expF1_lidar_sum \
  bash scripts/03_train_baseline.sh
```

预期:18-24h on 8×H20,AMOTA ~0.55-0.60(对比 baseline 0.442 涨 +0.10-0.15)

### Step 2:F1+F2 (MAFS)

```bash
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/phase2/expF1_F2_mafs.py \
  WORK_DIR=work_dirs/expF1_F2_mafs \
  bash scripts/03_train_baseline.sh
```

预期:18-24h on 8×H20,AMOTA ~0.56-0.62(MAFS 比 sum 提 +0.01-0.02)

### Step 3:F_full (P2 SOTA)

```bash
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/phase2/expF_full.py \
  WORK_DIR=work_dirs/expF_full \
  bash scripts/03_train_baseline.sh
```

预期:18-24h on 8×H20,AMOTA ~0.65-0.70(双模态推理)

### Step 4:F_full ckpt 跑 3 种推理评测(免费,纯 evaluation)

#### 4a. 双模态推理(默认)

```bash
USE_GLOBAL_PYTHON=1 GPUS=8 \
  bash scripts/04_eval_my_baseline.sh work_dirs/expF_full/latest.pth
```

#### 4b. Cam-only 推理(对标 DualViewDistill)

需要在评测时 zero-out lidar feature。两种做法:
- **简单**:把 expF_full.py 复制为 expF_full_cam_only_eval.py,加 `data['points']=None` patch
- **优雅**:加 eval_mode env 在 sparse4d.py 里识别(暂未实现,后续补)

#### 4c. LiDAR-only 推理(对标 CenterPoint)

类似 4b,但 zero-out image feature。

## Phase 2 完整 ablation table 模板

| Method | Mod (train) | Mod (infer) | mAP | NDS | AMOTA |
|---|---|---|---|---|---|
| Baseline (Sparse4D-v3, R50) | Cam | Cam | 0.449 | 0.559 | 0.442 |
| **F1 LiDAR + sum fusion** | Cam+Lidar | Cam+Lidar | ? | ? | ~0.55-0.60 |
| **F1+F2 MAFS** | Cam+Lidar | Cam+Lidar | ? | ? | ~0.56-0.62 |
| **F_full = F1+F2+F4** | Cam+Lidar | Cam+Lidar | ? | ? | **~0.65-0.70** |
| F_full (cam-only inference) | Cam+Lidar | Cam | ? | ? | ~0.60-0.65 |
| F_full (lidar-only inference) | Cam+Lidar | Lidar | ? | ? | ~0.65-0.70 |

## 显存预估

baseline (camera-only) 14.6 GB。LiDAR backbone 加上后:
- F1 sum:~20-22 GB(samples_per_gpu=6 仍 OK)
- F2 MAFS:~20-22 GB
- F_full:~20-22 GB(F4 masked-modal 不增加显存)

H20 总 97 GB,**完全 OK**。如果在 V100 32GB 上跑,需要把 `samples_per_gpu` 从 6 降到 4。

## 注意事项

1. **LiDAR backbone 第一次 forward 编译慢**:SparseEncoder 用 spconv,JIT 编译 GPU kernel,
   第 1 个 iter 可能 5-10 秒。第 2 个 iter 起恢复 ~1 s/iter。
2. **dataloader 可能 CPU bound**:LiDAR voxelize 在 CPU 上做,建议 `workers_per_gpu=8` 或更高
   (我们机器有 192 个 CPU core,设 16 也行)。
3. **第一帧 scene 起点 sweeps len=0**:`pad_empty_sweeps=True` 会用当前帧 padding,
   避免不同 sample 点云规模差太多导致 voxel layer max_voxels 不一致。
