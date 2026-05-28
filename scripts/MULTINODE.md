# 单机/多机自适应训练脚本 — `scripts/05_train_multinode.sh`

> 一份脚本同时支持**单机 8 卡**和**多机调度平台多机多卡**(2 机 16 卡 / 20 机 160 卡都行),用户**只需要设 CONFIG 一个 env var**。

## 1. 设计原则

- **平台调度兼容**:k8s 调度型平台启动 pod 时会**自动注入** `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` 等环境变量,脚本直接读取,无需手工配。
- **单机自动 fallback**:不在多机平台调度时(本地启动 / 单 8 卡机启动),所有节点 env var 默认 `NNODES=1, NODE_RANK=0, MASTER_ADDR=127.0.0.1`,等价于单机 `torchrun --nproc_per_node=8`。
- **WORK_DIR 自动派生**:从 `CONFIG` 文件名推出,e.g. `expF_full.py` → `work_dirs/expF_full`。
- **NCCL 跨机通信预调优**:Mellanox bond IB 网卡 + 多机 H20 的标准 env var 已 hardcode(从 `/mnt/volumes/ad-perception-al-sh01/dyf/onemodel` 工程实战值借),用户无需关心。

## 2. 启动方式

### 2.1 单机 8 卡(开发机 / 单训练机)

```bash
CONFIG=projects/configs/ablations/phase2/expF1_lidar_sum.py \
  bash scripts/05_train_multinode.sh
```

**等价于**:

```bash
USE_GLOBAL_PYTHON=1 \
  CONFIG=projects/configs/ablations/phase2/expF1_lidar_sum.py \
  WORK_DIR=work_dirs/expF1_lidar_sum \
  bash scripts/03_train_baseline.sh
```

但 `05_train_multinode.sh` 用 `torchrun`(替代 deprecated 的 `torch.distributed.launch`),更稳健。

### 2.2 多机调度平台(k8s,20 机 160 卡)

在调度平台界面填启动命令,平台会**自动**给每个 pod 注入 `NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT`:

```bash
CONFIG=projects/configs/ablations/phase2/expF_full.py \
  bash scripts/05_train_multinode.sh
```

平台调度配置示例(`感知 Onemodel · H20 · 上海01` 队列):
- 资源规格:八卡 H20-141 (150 核,1.6T 内存)
- 节点配置:Master × 1, Worker × 19 = 20 节点 × 8 卡 = **160 GPU**

### 2.3 手工跨机(没有调度平台时,2 机调试场景)

需要在每台机器上分别启动一份脚本,显式设环境变量:

**机器 A(master,IP 10.0.0.1)**:
```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  CONFIG=projects/configs/ablations/phase2/expF_full.py \
  bash scripts/05_train_multinode.sh
```

**机器 B(worker)**:
```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  CONFIG=projects/configs/ablations/phase2/expF_full.py \
  bash scripts/05_train_multinode.sh
```

## 3. 参数说明

| Env var | 必填 | 默认值 | 含义 |
|---|---|---|---|
| `CONFIG` | ✅ 必填 | — | mmcv config 文件路径 |
| `WORK_DIR` | 选 | `work_dirs/<config_name>` | 输出 ckpt + log 目录 |
| `NNODES` | 选(平台注入) | `1` 或 `WORLD_SIZE` | 总节点数 |
| `NODE_RANK` | 选(平台注入) | `0` 或 `RANK` | 本节点 rank |
| `MASTER_ADDR` | 选(平台注入) | `127.0.0.1` | rendezvous 地址 |
| `MASTER_PORT` | 选(平台注入) | `29500` | rendezvous 端口 |
| `GPUS_PER_NODE` | 选 | `8` | 每节点 GPU 数 |
| `CUDA_VISIBLE_DEVICES` | 选 | `0,1,...,GPUS_PER_NODE-1` | 用哪些卡 |
| `USE_GLOBAL_PYTHON` | 选 | `1`(镜像模式) | 走系统 python3.11 |
| 其他 NCCL_* / TORCH_NCCL_* | 选 | hardcode 默认 | IB 网络调优,默认值已实战验证 |

## 4. 多机训练前必须知道的事(批量训练前看这里)

### 4.1 LR auto-scaling — **脚本自动做,你不用手动改 config!**

baseline `samples_per_gpu=6`:
- 单机 8 卡 → `total_batch_size = 6 × 8 = 48`,baseline lr = `6e-4`
- 20 机 160 卡 → `total_batch_size = 6 × 160 = 960`(放大 **20×**),lr **必须重调**

**脚本自动 scale lr**(默认 sqrt 策略):
```
TOTAL_GPUS         scale_ratio    sqrt scaling lr     linear scaling lr
8 卡 (baseline)    1×             6.0000e-4 (不动)    6.0000e-4
16 卡 (2 机)       2×             8.4853e-4           1.2000e-3
32 卡 (4 机)       4×             1.2000e-3           2.4000e-3
64 卡 (8 机)       8×             1.6971e-3           4.8000e-3
160 卡 (20 机)     20×            2.6833e-3           1.2000e-2  ← 太激进易爆
```

启动后 echo 会显示当前 effective lr:
```
AUTO LR-SCALING = sqrt scaling: lr 6.000e-04 × 4.4721 = 2.683e-03 (TOTAL_GPUS/8 = 20.0000×)
```

**默认 sqrt scaling**,适合 large-batch (≥ 256) 场景,稳健不爆梯度。  
如果想换策略:
```bash
LR_SCALING=linear CONFIG=...  bash scripts/05_train_multinode.sh   # linear scaling(激进)
LR_SCALING=none   CONFIG=...  bash scripts/05_train_multinode.sh   # 禁用 auto-scaling,完全手动调
```

**实现细节**:脚本用 `mmcv.Config.fromfile(CONFIG)` 解析 baseline lr,按 `LR_SCALING` 策略计算 scaled lr,通过 `--cfg-options optimizer.lr=...` 在启动 train.py 时**动态注入**,完全不修改 config 文件。

**注意**:warmup_iters **不 scale**(因为 large batch 收敛快,不需要更长 warmup)。如果第一个 epoch 内 grad_norm > clip_thr=25 频繁触发 → 改 LR_SCALING=none 自己手动调,或在 config 里加大 `warmup_iters`。

### 4.2 强烈建议先 smoke 验证再上规模

`sqrt scaling` 在 nuScenes Sparse4D 上**没有官方实证数据**(论文只用 1 机 8 卡)。多机训练前**强烈建议**:
1. 先 2 机 16 卡 smoke 跑 ~500 iter,看 grad_norm 是否健康(< 100)
2. 如果 OK,直接跑 20 机 160 卡完整训练
3. 如果 grad_norm > 1000(梯度爆炸),改 `LR_SCALING=none` 自己手动调小 lr

### 4.3 自动 iter scaling — **总 epoch 不变,iter 数自动减少**

baseline 主 config 里 `num_iters_per_epoch=586` 用 `num_gpus=8` hardcode 算的:

```python
total_batch_size = 48
num_gpus = 8
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))  # = 586
num_epochs = 100
runner = dict(max_iters=num_iters_per_epoch * num_epochs)    # = 58600
```

多机时 mmcv **不会自动重算** `num_iters_per_epoch`,如果不修 max_iters,实际会跑 `100 × N` epoch(过训!)。

**脚本自动 scale iter 数**(不管 LR_SCALING 设置):
```
TOTAL_GPUS    max_iters             eval/ckpt_interval     总 epoch
8 卡          58600 (不变)          11720 (不变)           100
16 卡 (2 机)  58600 → 29300         11720 → 5860           100
32 卡 (4 机)  58600 → 14650         11720 → 2930           100
64 卡 (8 机)  58600 → 7325          11720 → 1465           100
160 卡 (20 机) 58600 → 2930         11720 → 586            100
```

启动后 echo 显示:
```
AUTO ITER-SCALE = 100 epoch unchanged. max_iters 58600→2930, eval_interval 11720→586, ckpt_interval 11720→586
```

**所以 `expF_full.py` 的 epoch 数永远是 100**(从 baseline `num_epochs=100` 来的),无论你用几台机器。脚本通过 `--cfg-options runner.max_iters=...` 注入,**不修改 config 文件**。

### 4.2 iter 数自动减少(epoch 数不变)

mmcv 的 `IterBasedRunner` 一个 epoch 跑 `len(dataset) / total_batch_size` 个 iter。多机后 total_batch_size 放大,iter 数线性减少:
- 单机 8 卡:每 epoch 586 iter,100 epoch 共 58600 iter
- 20 机 160 卡:每 epoch ~30 iter,100 epoch 共 ~3000 iter

config 里 `num_iters_per_epoch / num_epochs` 字段如果是手算的,要相应调整。

### 4.3 nuScenes scene 数限制

Sparse4D 用 `with_seq_flag=True, sequences_split_num=2`,把 scene 切成 2 段,每张卡拿完整子序列。850 scenes:
- 单机 8 卡:每卡 ~212 scene
- 20 机 160 卡:每卡 ~10 scene

160 卡时每卡 scene 数已经接近极限,**实际效率不会线性 scale**(可能只快 5-8 倍而不是 20 倍)。Phase 2(模型大、训练时间长)更值得上多机,Phase 1(R50 模型小)上多机 ROI 一般。

### 4.4 通信开销

NVLink(单机内部)~600 GB/s,IB(跨机)~200-400 Gb/s,慢 1-2 个数量级。**模型越小通信占比越高**:
- Sparse4D R50 ~48 MB params,单次 AllReduce 传输不大,但 head 中间 tensor 多
- 多机训练里 ~20-30% 时间花在跨机通信上(实测 OneModel 工程数据)

## 5. 启动后看哪里确认 OK

### 5.1 stdout 头部应该看到

```
================================================================
>>> Sparse4D Multi-Node Training
================================================================
  CONFIG          = projects/configs/ablations/phase2/expF_full.py
  WORK_DIR        = work_dirs/expF_full
  NNODES          = 20
  NODE_RANK       = 0
  GPUS_PER_NODE   = 8
  TOTAL_GPUS      = 160
  MASTER_ADDR     = 10.168.204.198
  ...
================================================================
```

NNODES 显示对(如 20)= 平台调度成功;显示 1 = 单机或调度失败。

### 5.2 第一个 iter 之后

正常 mmcv runner log:
```
mmdet - INFO - workflow: [('train', 1)], max: <iter数> iters
mmdet - INFO - Iter [51/<iter数>] lr: ... loss: ... grad_norm: ...
```

异常信号:
- `RuntimeError: connect timeout` / `NCCL ... timeout` → IB 配置问题或 MASTER_ADDR 错
- 长时间不动 → 检查所有 NODE_RANK 都连上(看 master 节点 log 里 `Worker connected from ...`)

### 5.3 Grafana 监控关键 panel

- **GPU 利用率**:每节点应该 > 90%(< 70% 说明 IO/CPU bound)
- **宿主机 RDMA 网卡速率**:多机时跨机 AllReduce 应该看到 IB 流量
- **PFC Pause frame latency**:IB 网络拥塞时会上涨,> 100ms 说明跨机性能差
- **XID ERROR**:一直 0 才正常,任何 spike 都说明硬件故障

## 6. 常见错误排查

### 6.1 `RuntimeError: connect timeout` 跨机连接失败

**根因**:平台没正确注入 `MASTER_ADDR`,或 IB 网络配置错。
**修法**:
```bash
# 在 worker 节点测能否 ping master
ping ${MASTER_ADDR}
# 测端口
telnet ${MASTER_ADDR} ${MASTER_PORT}
```
端口不通就是平台 firewall / network policy 问题,需要联系运维。

### 6.2 `find_unused_parameters` 错误

某些 trick(如 P1 PFTrack 的 motion_mlp)在第一个 iter 不参与 forward,DDP 报错。
**修法**:在 config 加 `find_unused_parameters=True`(注意性能略降)。

### 6.3 OOM(显存溢出)

P2 有 LiDAR backbone,显存比 P0 baseline 多 ~5-7 GB。如果 H20 之外的卡(如 V100 32GB):
- 把 `samples_per_gpu` 从 6 降到 4(改 config)
- lr / warmup / iter 同步 scale

### 6.4 `lessons-learned.mdc` 已记录的多机相关坑

回看 `.cursor/rules/lessons-learned.mdc`,踩过的坑都在那。

## 7. 常用启动命令速查

```bash
# Phase 0 baseline 重训
CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
  bash scripts/05_train_multinode.sh

# Phase 2 F1
CONFIG=projects/configs/ablations/phase2/expF1_lidar_sum.py \
  bash scripts/05_train_multinode.sh

# Phase 2 F1+F2 MAFS
CONFIG=projects/configs/ablations/phase2/expF1_F2_mafs.py \
  bash scripts/05_train_multinode.sh

# Phase 2 F_full(SOTA)
CONFIG=projects/configs/ablations/phase2/expF_full.py \
  bash scripts/05_train_multinode.sh

# 自定义 work_dir(避免覆盖之前 ckpt)
CONFIG=projects/configs/ablations/phase2/expF_full.py \
  WORK_DIR=work_dirs/expF_full_v2 \
  bash scripts/05_train_multinode.sh

# 单机调试 4 卡(GPUS_PER_NODE=4)
GPUS_PER_NODE=4 \
  CONFIG=projects/configs/ablations/phase2/expF1_lidar_sum.py \
  bash scripts/05_train_multinode.sh
```
