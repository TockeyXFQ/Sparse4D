#!/bin/bash
# =============================================================================
# 训练 Sparse4Dv3 (默认 R50 baseline,可通过 CONFIG / WORK_DIR env var override)
#
# 用法:
#   # 默认:训 baseline (8 卡 H20, 100 epoch, ~12-18h)
#   bash scripts/03_train_baseline.sh
#
#   # 训 Phase 1 ablation (CONFIG override)
#   USE_GLOBAL_PYTHON=1 \
#     CONFIG=projects/configs/ablations/expA_full_no_refine2d.py \
#     WORK_DIR=work_dirs/expA_full_no_refine2d \
#     bash scripts/03_train_baseline.sh
#
#   # 1 卡 smoke test (GPUS=1 + bs override via config)
#   GPUS=1 bash scripts/03_train_baseline.sh
#
# Env vars:
#   USE_GLOBAL_PYTHON  : 1=系统 python3.11 (镜像模式),其他=venv
#   CONFIG             : config 路径,默认 projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
#   WORK_DIR           : 输出目录,默认 work_dirs/baseline_v3_r50_repro
#   GPUS               : 卡数,默认 8(只支持 1 和 8)
#   PORT               : DDP master port,默认 28650
#
# Baseline 期望产出(作者数字,对应默认 config):
#     NDS    0.5636    mAP    0.4647
#     AMOTA  0.477     AMOTP  1.136     IDS    441
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "${SPARSE4D_ROOT}"

# ----- Python 环境选择 (3 种模式,优先级从高到低) -----
# 1. USE_GLOBAL_PYTHON=1  → 系统 /usr/local/bin/python3.11(镜像模式,推荐)
# 2. SPARSE4D_VENV=<path> → 指定 venv 路径(默认 /opt/sparse4d_env,镜像友好)
# 3. ${SPARSE4D_ROOT}/.venv → fallback 到开发机本地 venv
if [ "${USE_GLOBAL_PYTHON:-0}" = "1" ]; then
    export PATH=/usr/local/bin:${PATH:-}
    echo ">>> Using system python: $(which python3)"
else
    SPARSE4D_VENV=${SPARSE4D_VENV:-/opt/sparse4d_env}
    [ -d "${SPARSE4D_VENV}" ] || SPARSE4D_VENV="${SPARSE4D_ROOT}/.venv"
    if [ -d "${SPARSE4D_VENV}" ]; then
        # shellcheck disable=SC1091
        source "${SPARSE4D_VENV}/bin/activate"
        echo ">>> Using venv: ${SPARSE4D_VENV}"
    else
        export PATH=/usr/local/bin:${PATH:-}
        echo "[WARN] No venv at ${SPARSE4D_VENV}, falling back to system python: $(which python3)"
    fi
fi

# GPUS: 默认 8(训练机),开发机 smoke test 用 GPUS=1
GPUS=${GPUS:-8}
case "${GPUS}" in
    1) export CUDA_VISIBLE_DEVICES=0 ;;
    8) export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    *) echo "[FAIL] GPUS=${GPUS} not supported (only 1 or 8)"; exit 1 ;;
esac
export PYTHONPATH=${PWD}:${PYTHONPATH:-}
export PORT=${PORT:-28650}

# H20 性能调优(可选)
export OMP_NUM_THREADS=8

# CONFIG / WORK_DIR 都可以通过 env var override(便于跑 ablation)
CONFIG=${CONFIG:-projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py}
WORK_DIR=${WORK_DIR:-work_dirs/baseline_v3_r50_repro}

if [ ! -f "${CONFIG}" ]; then
    echo "[FAIL] config not found: ${CONFIG}"
    exit 1
fi

mkdir -p "${WORK_DIR}"

echo ">>> Training Sparse4Dv3 on ${GPUS} GPU(s)"
echo "    config   = ${CONFIG}"
echo "    work dir = ${WORK_DIR}"

bash tools/dist_train.sh \
    "${CONFIG}" \
    "${GPUS}" \
    --work-dir "${WORK_DIR}" \
    2>&1 | tee -a "${WORK_DIR}/train.log"
