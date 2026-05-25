#!/bin/bash
# =============================================================================
# 训练 Sparse4Dv3 R50 baseline (8 卡 H20, 100 epoch, 严格对齐论文 config)
# 期望时间: ~12-18 小时 (8 卡 H20, 比 RTX 3090 快 ~1.5 倍)
# 期望产出 (作者数字):
#     NDS    0.5636    mAP    0.4647
#     AMOTA  0.477     AMOTP  1.136     IDS    441
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "${SPARSE4D_ROOT}"

# shellcheck disable=SC1091
source "${SPARSE4D_ROOT}/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=${PWD}:${PYTHONPATH:-}
export PORT=28650

# H20 性能调优(可选)
export OMP_NUM_THREADS=8

CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
WORK_DIR=work_dirs/baseline_v3_r50_repro

mkdir -p "${WORK_DIR}"

echo ">>> Training Sparse4Dv3 R50 baseline on 8 GPUs"
echo "    config   = ${CONFIG}"
echo "    work dir = ${WORK_DIR}"
echo "    epochs   = 100  (~12-18h on 8x H20)"

bash tools/dist_train.sh \
    "${CONFIG}" \
    8 \
    --work-dir "${WORK_DIR}" \
    2>&1 | tee -a "${WORK_DIR}/train.log"
