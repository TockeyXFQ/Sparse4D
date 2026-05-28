#!/bin/bash
# =============================================================================
# 单机/多机自适应训练脚本(Sparse4D)
#
# 用户只需设 CONFIG env var,其他全部自动:
#   - 单机时(NNODES=1):等价于 8 卡 DDP,跟 03_train_baseline.sh 一致
#   - 多机时(NNODES>1):自动启动 torchrun 多机 rendezvous,无需手工配
#                       MASTER_ADDR / NODE_RANK 等(平台调度时自动注入)
#
# 必填:
#   CONFIG  — config 文件路径,e.g. projects/configs/ablations/phase2/expF_full.py
#
# 选填(平台多机调度时通常自动注入):
#   NNODES         总节点数(默认 1)
#   NODE_RANK      本节点 rank(默认 0)
#   MASTER_ADDR    rendezvous 地址(默认 127.0.0.1)
#   MASTER_PORT    rendezvous 端口(默认 29500)
#   GPUS_PER_NODE  每节点 GPU 数(默认 8)
#   WORK_DIR       输出目录(默认 work_dirs/<config_name>)
#   USE_GLOBAL_PYTHON=1  走系统 python3.11(镜像模式;默认 1)
#
# 启动示例:
#   单机 8 卡:
#     CONFIG=projects/configs/ablations/phase2/expF_full.py \
#       bash scripts/05_train_multinode.sh
#
#   k8s 平台多机调度(20 机 160 卡)— 平台启动命令模板:
#     CONFIG=projects/configs/ablations/phase2/expF_full.py \
#       bash scripts/05_train_multinode.sh
#     (NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT 由平台自动注入)
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "${SPARSE4D_ROOT}"

# ----- 必填参数检查 -----
if [ -z "${CONFIG:-}" ]; then
    echo "[FAIL] CONFIG env var is required."
    echo "Example:"
    echo "  CONFIG=projects/configs/ablations/phase2/expF_full.py \\"
    echo "    bash scripts/05_train_multinode.sh"
    exit 1
fi
if [ ! -f "${CONFIG}" ]; then
    echo "[FAIL] config file not found: ${CONFIG}"
    exit 1
fi

# ----- Python 环境(默认镜像模式,与 03_train_baseline.sh 一致)-----
USE_GLOBAL_PYTHON=${USE_GLOBAL_PYTHON:-1}
if [ "${USE_GLOBAL_PYTHON}" = "1" ]; then
    export PATH=/usr/local/bin:${PATH:-}
else
    SPARSE4D_VENV=${SPARSE4D_VENV:-/opt/sparse4d_env}
    [ -d "${SPARSE4D_VENV}" ] || SPARSE4D_VENV="${SPARSE4D_ROOT}/.venv"
    if [ -d "${SPARSE4D_VENV}" ]; then
        # shellcheck disable=SC1091
        source "${SPARSE4D_VENV}/bin/activate"
    else
        export PATH=/usr/local/bin:${PATH:-}
    fi
fi

# ----- NCCL / IB 跨机通信调优(参考同集群 onemodel/dyf 工程实战值)-----
# 这些 env var 的取值已经在我们的训练集群被反复验证过(Mellanox bond IB 网卡 +
# 多机 H20),直接复用避免每次踩坑。如果换集群,这些可能需要调整,但单机训练
# 完全没影响(NCCL 单进程 NVLink 通信不走 IB)。
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TC=${NCCL_IB_TC:-136}
export NCCL_IB_SL=${NCCL_IB_SL:-5}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
export NCCL_IB_SPLIT_DATA_ON_QPS=${NCCL_IB_SPLIT_DATA_ON_QPS:-1}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# ----- 节点配置自适应(单机/多机一份脚本搞定)-----
# 兼容多种平台环境变量命名约定,优先级:
#   NNODES > WORLD_SIZE > 1
#   NODE_RANK > RANK > 0
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# CUDA_VISIBLE_DEVICES 默认用 0..GPUS_PER_NODE-1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
fi
export PYTHONPATH=${PWD}:${PYTHONPATH:-}

# ----- WORK_DIR 默认从 config 名字派生 -----
_CFG_TAG=$(basename "${CONFIG}" .py)
WORK_DIR=${WORK_DIR:-work_dirs/${_CFG_TAG}}
mkdir -p "${WORK_DIR}"

# ----- 自动 LR Scaling(根据 TOTAL_GPUS 自动调 lr / warmup_iters)-----
# 当 TOTAL_GPUS != BASELINE_GPUS 时,自动按 LR_SCALING 策略 scale lr,通过
# mmcv --cfg-options 动态注入,**不修改 config 文件**。warmup_iters 同步 scale。
#
# 策略选择:
#   sqrt   (默认,推荐):scaled_lr = baseline_lr × sqrt(TOTAL_GPUS / BASELINE_GPUS)
#                                    适合 large-batch (≥ 256) 场景,稳健
#   linear (激进):     scaled_lr = baseline_lr × (TOTAL_GPUS / BASELINE_GPUS)
#                                    适合 small-batch (< 256),lr 涨速快
#   none   (禁用):     不 scale,用 config 原值(用户手动调时设此项)
#
# warmup_iters 用相同 factor scale,保证 warmup 占比合理:
#   单机 8 卡:  warmup_iters=500 / total=58600 = 0.85%(原值)
#   20 机 160 卡 sqrt: warmup_iters=500×√20 ≈ 2236 / total=2930 = 76% (太长)
#   实际只 scale lr,warmup_iters 保持原值(因为 large batch 收敛快不需要更长
#   warmup,只需要 lr 不爆;若 grad_norm 频繁 clip,改 LR_SCALING=none 自调)
LR_SCALING=${LR_SCALING:-sqrt}
BASELINE_GPUS=${BASELINE_GPUS:-8}

EXTRA_OPTS=()
LR_INFO=""
ITER_INFO=""

# 一次 python call 拿到 baseline_lr / num_iters_per_epoch / num_epochs / max_iters
# / evaluation.interval / checkpoint.interval,然后:
#   1. lr 按 LR_SCALING 策略 scale
#   2. iter 数按 TOTAL_GPUS / BASELINE_GPUS scale(防多机过训)
SCALED_RESULT=$(PYTHONPATH=${PYTHONPATH} python3 - <<PY 2>/dev/null
import math, warnings
warnings.filterwarnings('ignore')
from mmcv import Config
c = Config.fromfile('${CONFIG}')

baseline_lr = float(c.optimizer.lr)
total_gpus = ${TOTAL_GPUS}
baseline_gpus = ${BASELINE_GPUS}
scale_ratio = total_gpus / baseline_gpus
strategy = '${LR_SCALING}'

# ---- LR scaling factor ----
if strategy == 'sqrt':
    factor = math.sqrt(scale_ratio)
elif strategy == 'linear':
    factor = scale_ratio
else:  # 'none'
    factor = 1.0
scaled_lr = baseline_lr * factor

# ---- Iter scaling: 总 sample 数固定,batch 大 N× → iter 减 N× ----
# baseline 主 config 里 num_iters_per_epoch 用 num_gpus=8 hardcode 算的,
# 多机时不会自动重算。这里手动 scale runner.max_iters / evaluation.interval
# / checkpoint_config.interval 三个字段,保证总 epoch 数不变。
baseline_max_iters = int(c.runner.max_iters)
baseline_eval_interval = int(c.evaluation.interval)
baseline_ckpt_interval = int(c.checkpoint_config.interval)
# scale 1/N (向下取整,避免最后超出)
scaled_max_iters = max(int(baseline_max_iters / scale_ratio), 1)
scaled_eval_interval = max(int(baseline_eval_interval / scale_ratio), 1)
scaled_ckpt_interval = max(int(baseline_ckpt_interval / scale_ratio), 1)

# 也输出 num_epochs 对应估计(用主 config 已 hardcode 的 num_iters_per_epoch
# 反推,但实际 num_epochs 是 baseline_max_iters / num_iters_per_epoch)
try:
    num_epochs = int(getattr(c, 'num_epochs', 100))
except Exception:
    num_epochs = baseline_max_iters // 586  # fallback

print(f'{baseline_lr:.6e} {scaled_lr:.6e} {scale_ratio:.4f} {factor:.4f} '
      f'{baseline_max_iters} {scaled_max_iters} '
      f'{baseline_eval_interval} {scaled_eval_interval} '
      f'{baseline_ckpt_interval} {scaled_ckpt_interval} {num_epochs}')
PY
)

if [ -n "${SCALED_RESULT}" ]; then
    read -r BASELINE_LR SCALED_LR SCALE_RATIO FACTOR \
            BASELINE_MAX_ITERS SCALED_MAX_ITERS \
            BASELINE_EVAL_INTERVAL SCALED_EVAL_INTERVAL \
            BASELINE_CKPT_INTERVAL SCALED_CKPT_INTERVAL \
            NUM_EPOCHS <<< "${SCALED_RESULT}"

    if [ "${LR_SCALING}" != "none" ] && [ "${TOTAL_GPUS}" != "${BASELINE_GPUS}" ]; then
        EXTRA_OPTS+=("--cfg-options" "optimizer.lr=${SCALED_LR}")
        LR_INFO="${LR_SCALING}: lr ${BASELINE_LR} × ${FACTOR} = ${SCALED_LR} (${SCALE_RATIO}×)"
    else
        LR_INFO="not scaling (TOTAL_GPUS=${TOTAL_GPUS} == BASELINE_GPUS=${BASELINE_GPUS}, or LR_SCALING=none)"
    fi

    # iter 数永远 scale(不管 LR_SCALING 设置),否则多机会过训 N×
    if [ "${TOTAL_GPUS}" != "${BASELINE_GPUS}" ]; then
        EXTRA_OPTS+=(
            "runner.max_iters=${SCALED_MAX_ITERS}"
            "evaluation.interval=${SCALED_EVAL_INTERVAL}"
            "checkpoint_config.interval=${SCALED_CKPT_INTERVAL}"
        )
        ITER_INFO="${NUM_EPOCHS} epoch unchanged. max_iters ${BASELINE_MAX_ITERS}→${SCALED_MAX_ITERS}, eval_interval ${BASELINE_EVAL_INTERVAL}→${SCALED_EVAL_INTERVAL}, ckpt_interval ${BASELINE_CKPT_INTERVAL}→${SCALED_CKPT_INTERVAL}"
    else
        ITER_INFO="${NUM_EPOCHS} epoch (${BASELINE_MAX_ITERS} iter, no scaling needed)"
    fi
else
    LR_INFO="WARN: failed to parse ${CONFIG}, NOT auto-scaling lr / iter"
    ITER_INFO=""
fi

# ----- 启动信息 -----
echo "================================================================="
echo ">>> Sparse4D Multi-Node Training"
echo "================================================================="
echo "  CONFIG          = ${CONFIG}"
echo "  WORK_DIR        = ${WORK_DIR}"
echo "  NNODES          = ${NNODES}"
echo "  NODE_RANK       = ${NODE_RANK}"
echo "  GPUS_PER_NODE   = ${GPUS_PER_NODE}"
echo "  TOTAL_GPUS      = ${TOTAL_GPUS}"
echo "  MASTER_ADDR     = ${MASTER_ADDR}"
echo "  MASTER_PORT     = ${MASTER_PORT}"
echo "  CUDA_VISIBLE    = ${CUDA_VISIBLE_DEVICES}"
echo "  Python          = $(which python3) ($(python3 --version 2>&1 | head -1))"
echo "  AUTO LR-SCALING = ${LR_INFO}"
if [ -n "${ITER_INFO}" ]; then
    echo "  AUTO ITER-SCALE = ${ITER_INFO}"
fi
echo "================================================================="

# ----- 启动 torchrun(替代 deprecated 的 torch.distributed.launch)-----
# torchrun 的 --rdzv-* 参数让多机 rendezvous 更稳健;单机时也能跑(等价
# torch.distributed.launch)。
torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    tools/train.py \
    "${CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    "${EXTRA_OPTS[@]}" \
    "$@" \
    2>&1 | tee -a "${WORK_DIR}/train.log"
