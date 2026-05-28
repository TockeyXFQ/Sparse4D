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
    "$@" \
    2>&1 | tee -a "${WORK_DIR}/train.log"
