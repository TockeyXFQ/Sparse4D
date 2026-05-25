# Sparse4Dv3 R50 Baseline 复现指南(8 卡 H20)

> **目标**:在本机 8 卡 H20(系统 CUDA 12.1,Python 3.10)上完成
> - Step 0: 环境配置
> - Step 1: 数据 + 预训练权重准备
> - Step 2: 用作者放出的 ckpt 跑评测 → **对齐 evaluation pipeline**
> - Step 3: 训练自己的 baseline(100 epoch,~12-18h)
> - Step 4: 评测自己训出的 baseline
>
> **所有命令都用 8 卡(`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`),如有变化请改 `scripts/*.sh` 里的对应行。**

---

## 文件结构

```
Sparse4D-V3/
├── scripts/                          ← 全部新增
│   ├── 00_setup_env.sh               # 配环境(venv + torch + mmcv + mmdet + CUDA op)
│   ├── 01_prepare_data.sh            # 数据软链 + 下载 ckpt + 生成 pkl + 生成 anchor
│   ├── 02_eval_pretrained.sh         # 用作者 ckpt 跑评测(对齐 pipeline)
│   ├── 03_train_baseline.sh          # 训练自己 baseline(100 epoch)
│   └── 04_eval_my_baseline.sh        # 评测自己的 ckpt
├── .venv/                            ← venv 装在这里(00 步生成)
├── data/
│   ├── nuscenes -> /mnt/datasets/nuscenes/v1.0.0   (符号链接,01 步生成)
│   └── nuscenes_anno_pkls/           (01 步生成,~2 GB)
├── ckpt/                             ← 01 步生成
│   ├── resnet50-19c8e357.pth         (R50 backbone 预训练)
│   └── sparse4dv3_r50.pth            (作者放出的完整 ckpt)
├── nuscenes_kmeans900.npy            ← 01 步生成(必须放在工作根部)
└── work_dirs/                        ← 训练 / 评测产出
    ├── eval_pretrained_<ts>/         (02 步产出)
    ├── baseline_v3_r50_repro/        (03 步产出: 训练 log + ckpt)
    └── eval_my_baseline_<ts>/        (04 步产出)
```

---

## Step 0 - 环境配置

```bash
cd /mnt/volumes/ad-perception-al-sh02/liuqingyu854/Track/Sparse4D-V3
bash scripts/00_setup_env.sh
```

**这个脚本做了 7 件事**:

| # | 动作 | 解释 |
|---|---|---|
| 1 | `/usr/bin/python3.10 -m venv .venv` | 用系统已有的 Python 3.10 建独立虚拟环境(Sparse4D 默认要 Python 3.8,但系统没装,3.10 实测可用) |
| 2 | `pip install --upgrade pip setuptools wheel` | 升级 pip 工具链到 23.3.2(避免老 pip 不识别 `--extra-index-url`) |
| 3 | `pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url ...` | **PyTorch 1.13 cu117 预编译版**。H20 是 Hopper sm_90,torch 1.13 wheel 不含 sm_90 SASS,但内嵌的 PTX 会**首次运行时被 driver JIT 编译**到 sm_90,可正常工作(略慢 5-10%,但稳定) |
| 4 | `pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html` | **mmcv-full 1.7.2 预编译版**,严格对齐 requirement.txt。`-f` 指定 OpenMMLab 的 wheel 索引 |
| 5 | `pip install mmdet==2.28.2 numpy==1.23.5 ...` | 其他严格按 requirement.txt 版本,补 `scikit-learn==1.3.2` 给 anchor_generator 用 |
| 6 | `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX" python setup.py develop`(在 `projects/mmdet3d_plugin/ops/` 下) | **重新编译 DFA CUDA op,显式包含 sm_90 PTX**,这是 H20 上能跑的关键 |
| 7 | 跑一个 Python 验收脚本,打印 torch / mmcv / mmdet / numpy 版本 + GPU compute capability + import 关键模块 | 确保整个环境装对 |

**选项含义说明**:
- `--extra-index-url`:除了 PyPI 默认源之外,**额外**加 PyTorch 官方 wheel 仓库(因为 cu117 wheel 不在 PyPI 上)
- `-f` 等价于 `--find-links`:把后面 URL 作为 wheel 索引页解析
- `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX"`:
  - `7.5` = T4 / RTX 2080;`8.0` = A100;`8.6` = A6000 / RTX 30;`9.0` = H100 / H20
  - `+PTX` 表示 9.0 不止生成 SASS,还生成 PTX(向上兼容下一代 GPU)
- `python setup.py develop`:**editable 安装**,源码改动立即生效,不用重装(后续做创新模块要改 op 时方便)

**预计时长**:**15-25 分钟**(pip 下载 + 编译 CUDA op)

**验收**:脚本最后会打印类似下面的输出,确认每个版本对齐:
```
============================================================
Python       : 3.10.12
torch        : 1.13.0+cu117
torch cuda   : 11.7
cuda visible : 8 GPU(s)
GPU name     : NVIDIA H20
compute cap  : (9, 0)
mmcv         : 1.7.2
mmdet        : 2.28.2
numpy        : 1.23.5
deformable_aggregation: OK
============================================================
```

如果 `compute cap` 不是 `(9, 0)` 或 `cuda visible` 不是 8,先排查 `nvidia-smi` 和 `CUDA_VISIBLE_DEVICES`。

---

## Step 1 - 数据 + 预训练权重 + pkl + anchor

```bash
cd /mnt/volumes/ad-perception-al-sh02/liuqingyu854/Track/Sparse4D-V3
bash scripts/01_prepare_data.sh
```

**这个脚本做了 5 件事**:

| # | 动作 | 解释 |
|---|---|---|
| 1 | `ln -s /mnt/datasets/nuscenes/v1.0.0 data/nuscenes` | 把 nuScenes 数据集软链到工作目录,**因为 config 里 `data_root = "data/nuscenes/"` 是相对路径** |
| 2 | 验证 `data/nuscenes/{samples, sweeps, v1.0-trainval, maps}` 4 个子目录都存在 | 缺一个都跑不了 |
| 3 | `wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth` | R50 ImageNet 预训练权重(98 MB),config 里 `img_backbone.pretrained` 指向此路径 |
| 4 | `wget https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth -O ckpt/sparse4dv3_r50.pth` | **作者放出的 Sparse4Dv3 R50 完整 ckpt(~550 MB)**,用来跑 Step 2 评测对齐 |
| 5 | `python tools/nuscenes_converter.py --root_path data/nuscenes --version v1.0-trainval --info_prefix data/nuscenes_anno_pkls/nuscenes --max_sweeps 10` | 把 nuScenes 元信息转成 Sparse4D 需要的 pkl 文件(train + val 两个),~20 分钟 |
| 6 | `python tools/anchor_generator.py --ann_file data/nuscenes_anno_pkls/nuscenes_infos_train.pkl --num_anchor 900 --detection_range 55 --output_file_name nuscenes_kmeans900.npy` | 在训练集 GT 中跑 k-means 生成 900 个 anchor 中心,~10 分钟 |

**选项含义说明**:
- `--root_path data/nuscenes`:nuScenes 数据根目录
- `--version v1.0-trainval`:用 trainval split(700 训 + 150 val)
- `--info_prefix`:输出 pkl 的路径前缀,会得到 `nuscenes_infos_train.pkl` + `nuscenes_infos_val.pkl`
- `--max_sweeps 10`:每帧关键帧之前再回溯 10 个 sweep(虽然 Sparse4Dv3 不用 sweep,但 converter 保留这个字段方便后续多模态扩展)
- `--num_anchor 900`:k-means 聚类数,**必须 = config 里的 `num_anchor=900`**
- `--detection_range 55`:只用 55m 内的 GT 聚类(对齐 config 里 `CircleObjectRangeFilter` 的 55 m)
- `--output_file_name nuscenes_kmeans900.npy`:**必须叫这个名,且放在工作目录根部**(config 里 `instance_bank.anchor="nuscenes_kmeans900.npy"` 是相对路径)

**预计时长**:**45-60 分钟**(主要花在 wget 下载和 pkl 生成上)

**验收**:脚本结尾会打印各文件大小,正常应有:
```
[OK] data/nuscenes -> /mnt/datasets/nuscenes/v1.0.0
[OK] nuScenes data root layout verified
[OK] ckpt/resnet50-19c8e357.pth (98M)
[OK] ckpt/sparse4dv3_r50.pth (550M)
[OK] pkl: train=600M, val=200M
[OK] anchor: 80K
```

---

## Step 2 - 对齐评测 pipeline(用作者 ckpt)

```bash
cd /mnt/volumes/ad-perception-al-sh02/liuqingyu854/Track/Sparse4D-V3
bash scripts/02_eval_pretrained.sh
```

**这个脚本就一个核心命令**:

```bash
bash tools/dist_test.sh \
    projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
    ckpt/sparse4dv3_r50.pth \
    8 \
    --eval bbox \
    --eval-options jsonfile_prefix=work_dirs/eval_pretrained_<ts>/results
```

**逐项解释**:
- 第 1 个位置参数 `projects/configs/...py`:config 路径
- 第 2 个位置参数 `ckpt/sparse4dv3_r50.pth`:要评测的 ckpt(这里是**作者放出的版本**)
- 第 3 个位置参数 `8`:GPU 数,会启动 8 个 `torch.distributed` 进程,均分 val set 加速评测
- `--eval bbox`:让 mmdet 调用 dataset 的 `evaluate()` 方法。**Sparse4D 自定义的 `NuScenes3DDetTrackDataset.evaluate()` 内部会同时跑 `for metric in ["detection", "tracking"]`**,所以 detection + tracking 一次就出
- `--eval-options jsonfile_prefix=...`:把模型输出的 nuScenes 标准 JSON 存到这里(后续可以再单独丢给 nuscenes-devkit 复算或可视化)

**Sparse4D 评测的内部机制**(`nuscenes_3d_det_track_dataset.py:578` 起):
- 一次推理后,模型输出的每个 box 都带 `boxes_3d, scores_3d, labels_3d`,**tracking 模式下还带 `instance_ids`**(由 `instance_bank.get_instance_id` 产出)
- `format_results` 把这些拼成 nuScenes 官方 JSON 格式(detection JSON + tracking JSON 两份)
- `_evaluate_single` 分别调用 `NuScenesEval`(detection)和 `TrackingEval`(tracking),官方 devkit 自动算 NDS / mAP / mATE / AMOTA / AMOTP / IDS / Recall 等所有指标
- **不需要再单独跑 `python -m nuscenes.eval.tracking.evaluate`**

**预计时长**:**25-40 分钟**(8 卡 H20 全速跑 6019 个 val sample)

**验收**:eval log 末尾会看到两段输出,**应当与 config 文件头部注释对齐**(误差 ±0.005):

期望输出(detection):
```
mAP: 0.4647
NDS: 0.5636
mATE: 0.5403, mASE: 0.2623, mAOE: 0.4590, mAVE: 0.2198, mAAE: 0.2059
```

期望输出(tracking):
```
AMOTA: 0.477
AMOTP: 1.136
IDS:   441
Recall: 0.588
```

如果你的数字在 ±0.005 内,说明 **环境 + 数据 + 评测 pipeline 全部对齐**,可以放心进 Step 3。

---

## Step 3 - 训练自己的 baseline

```bash
cd /mnt/volumes/ad-perception-al-sh02/liuqingyu854/Track/Sparse4D-V3

# 前台跑(可以看实时输出,但 ssh 断了就停)
bash scripts/03_train_baseline.sh

# 推荐:后台跑(用 nohup,日志写入 work_dirs/baseline_v3_r50_repro/train.log)
nohup bash scripts/03_train_baseline.sh > /dev/null 2>&1 &
echo "PID = $!"
```

**核心命令**:

```bash
bash tools/dist_train.sh \
    projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
    8 \
    --work-dir work_dirs/baseline_v3_r50_repro
```

**逐项解释**:
- 第 1 个位置参数 config 路径(完全沿用原 config,**不动任何超参**确保对齐论文)
- 第 2 个位置参数 `8`:8 卡分布式训练,对齐 config 里 `num_gpus=8, total_batch_size=48, samples_per_gpu=6`
- `--work-dir`:训练产出目录(每 20 epoch 存一次 ckpt,根据 `checkpoint_epoch_interval=20`,共存 5 个)

**对齐论文 config 的关键超参**(都已默认):
- `total_batch_size = 48`(8 GPU × batch 6)
- `num_epochs = 100`
- `num_iters_per_epoch = 28130 / 48 ≈ 585`,总 iter ≈ 58500
- `optimizer = AdamW(lr=6e-4, weight_decay=0.001)`,`img_backbone.lr_mult=0.5`
- `lr_config = CosineAnnealing`,warmup 500 iter
- `fp16, loss_scale=32`
- `IterBasedRunner`
- `evaluation.interval = 20 epoch × 585 iter`(每 20 epoch 验证一次)

**预计时长**:
- 原作者 8 卡 RTX 3090 用了 **22 小时**
- H20 单卡比 3090 单卡快约 2-3 倍(BF16/FP16 throughput)
- 估算 8 卡 H20:**12-18 小时**(取决于数据 IO 是否瓶颈)

**监控训练**:
```bash
# 实时看 log
tail -f work_dirs/baseline_v3_r50_repro/train.log

# 看 tensorboard(config 里默认开了 TensorboardLoggerHook)
tensorboard --logdir work_dirs/baseline_v3_r50_repro/tf_logs --port 6006 --bind_all
```

**关键观察点**(确保训练正常):
- 前 500 iter:`loss_cls + loss_box` 应从 ~10 快速下降到 ~3 以下(warmup 阶段)
- 1k iter 后:`loss` 应稳定下降,无 NaN
- 第 20 epoch (~11700 iter) 第一次 eval:**NDS 应 ≥ 0.50**(没到说明有问题)
- 第 60 epoch:**NDS 应 ≥ 0.54**
- 第 100 epoch:**NDS ≈ 0.5636, AMOTA ≈ 0.477**

**容灾**:
- 如果中断,加 `--resume-from work_dirs/baseline_v3_r50_repro/iter_xxxxx.pth` 续跑(改 `scripts/03_train_baseline.sh`)
- 如果显存爆(理论上 97G 不会,但万一):减 `samples_per_gpu`,但**会破坏对齐**

---

## Step 4 - 评测自己训出的 ckpt

训练结束后:

```bash
cd /mnt/volumes/ad-perception-al-sh02/liuqingyu854/Track/Sparse4D-V3

# 默认评测 latest.pth
bash scripts/04_eval_my_baseline.sh

# 或评测指定 ckpt
bash scripts/04_eval_my_baseline.sh work_dirs/baseline_v3_r50_repro/iter_58500.pth
```

**期望指标**(对齐 Step 2 的作者 ckpt,误差 ±0.005):

| 指标 | 期望值 | 我的复现 | 容差 |
|---|---|---|---|
| NDS | 0.5636 | ? | ±0.005 |
| mAP | 0.4647 | ? | ±0.005 |
| mATE | 0.5403 | ? | ±0.020 |
| AMOTA | 0.477 | ? | ±0.005 |
| AMOTP | 1.136 | ? | ±0.020 |
| IDS | 441 | ? | ±30 |

如果你训的数字在容差内 → **baseline 对齐 ✅,可以进 Phase 1 的改进实验**。
如果差距过大,常见原因:
- pkl 文件没用 `--max_sweeps 10`
- anchor 文件不是 `nuscenes_kmeans900.npy`(名字必须严格一致)
- 训练中断后续跑导致 LR schedule 错乱
- 8 卡之间负载不均衡(确认 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`)

---

## 一条龙命令(快速参考)

```bash
cd /mnt/volumes/ad-perception-al-sh02/liuqingyu854/Track/Sparse4D-V3

# 顺序跑(00 + 01 大约 1-1.5 小时)
bash scripts/00_setup_env.sh   && \
bash scripts/01_prepare_data.sh && \
bash scripts/02_eval_pretrained.sh

# 验证 Step 2 数字对齐后,启动训练(~12-18 小时)
nohup bash scripts/03_train_baseline.sh > /dev/null 2>&1 &

# 训完评测
bash scripts/04_eval_my_baseline.sh
```

---

## 后续(Phase 1 接入)

完成 baseline 复现后,**继续按 [`docs/research/sparse4dv3_improvement_plan.md`](../../docs/research/sparse4dv3_improvement_plan.md) 的 Phase 1 路线图**,逐个 trick 接入 ablation:

```
Phase 1 顺序:
  + B5 ContrasTR loss     (~半天调参 + 1 天训练)
  + B1 PF-Track STR       (~1-2 天接入 + 1 天训练)
  + B7 状态机             (~半天 + 重新评测)
  + D1 HQD                (~半天 + 1 天训练)
  + B3 UPD                (~1 天接入 + 1 天训练)
```

每个 trick 都通过 config flag 开关,**复制一份 config 改 flag 即可**,无需修改源码主干。
