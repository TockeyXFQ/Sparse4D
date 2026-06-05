# LiDAR BatchNorm 重标定 (`06_bn_recalibrate.py`)

## 解决什么问题

P2 的 LiDAR `SparseEncoder` 第一个 BN(`pts_middle_encoder.conv_input.1`)的
`running_mean/var` 在训练早期不稳定期被污染:

| ckpt | `conv_input.1.running_var` | 现象 |
|---|---|---|
| F1 (sum) | NaN | lidar-only 完全失效 |
| F_full (mafs) | ~1e17 | lidar-only AMOTA 仅 0.142 |

卷积权重本身是好的(无 NaN/Inf),**只是 BN 的两个非可学习 buffer 坏了**。eval
时 BN 用坏掉的 running stats 归一化 → LiDAR BEV 退化成与输入无关的常数图 →
lidar-only 检测崩溃。NaNStopHook 查的是 loss/grad,查不到 BN buffer,所以训练
全程"看起来健康"。

本脚本**只重估 LiDAR backbone 的 BN 统计量**(纯前向、不反向、不改任何可学习
权重),图像分支完全不动。

## 原理

1. 整个模型 `eval()`,然后**只**把 `pts_voxel_encoder / pts_middle_encoder /
   pts_backbone / pts_neck` 里的 BN 设成 train-mode 并 `reset_running_stats()`。
2. `momentum=None` → BN 用 cumulative moving average,跑 N 个 batch 后得到无偏
   平均的干净统计量。
3. 走 `simple_test` 前向(`self.training=False`,masked-modal 不触发),只有
   目标 BN 在 train-mode 会更新 running stats;图像 BN 保持 eval 不被污染。

## 用法

```bash
cd /mnt/volumes/ad-perception-al-sh01/liuqingyu/Sparse4D
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

python3 scripts/06_bn_recalibrate.py \
  projects/configs/ablations/phase2/expF_full.py \
  work_dirs/expF_full/iter_29300.pth \
  --num-batches 500 \
  --split train
```

跑完会在 `work_dirs/expF_full/iter_29300_bnrecalib.pth` 生成重标定后的 ckpt。

### 参数说明

| 参数 | 含义 | 默认 |
|---|---|---|
| `config`(位置) | config 文件路径 | 必填 |
| `checkpoint`(位置) | 待重标定的 ckpt | 必填 |
| `--out` | 输出 ckpt 路径 | 原 ckpt 加 `_bnrecalib` 后缀 |
| `--num-batches` | 重标定跑多少个 batch(BN 统计量收敛快,几百足够) | `500` |
| `--split` | 重标定数据来源:`train`(论文可用,避免碰 val)/ `val`(零兼容风险,快速诊断) | `train` |
| `--seed` | 随机种子 | `0` |
| `--cfg-options` | 覆盖 config 字段,多个 `key=value` 写在**同一个** `--cfg-options` 后 | 无 |

脚本会打印重标定**前后** `conv_input.1` 的 `running_var.max`,直接看到从 ~1e17
降到 O(1) 的改善。

## 重标定后:重测 G2 lidar-only 验证

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH
PORT=29516 bash tools/dist_test.sh \
  projects/configs/ablations/phase2/expF_full.py \
  work_dirs/expF_full/iter_29300_bnrecalib.pth 1 \
  --eval bbox \
  --eval-options jsonfile_prefix=work_dirs/expF_full/eval_G2_bnrecalib \
  --cfg-options model.eval_force_modal=lidar
```

- lidar-only AMOTA 若从 0.142 显著回升 → 确诊是 BN 损坏,且此 ckpt 被救回。
- 建议同时重测双模态(去掉 `model.eval_force_modal`)和 G1 cam-only,看 LiDAR
  在融合里的真实贡献。

## 注意

- 这是**纯前向重标定**,不更新任何可学习权重,不是重新训练。
- 若 `--split train` 加载报错(train ann 与 test pipeline 兼容性),改用
  `--split val` 先拿诊断结论。
- 这只是"路线 1"快速修复。根治需重训时修 LiDAR 输入归一化 + 给 mafs 融合加
  ReZero gate + BN momentum/SyncBN 调整(见对话里的路线 2)。
