#!/bin/bash
# =============================================================================
# 用作者放出的 Sparse4Dv3 R50 ckpt 跑 detection + tracking 双评测
# 目的: 对齐评测 pipeline,验证数据 + 评测脚本正常,无需训练
# 期望时间: ~30 分钟 (8 卡 H20)
# 期望指标 (config 注释里官方数字, val split):
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
export PORT=29512

CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
CKPT=ckpt/sparse4dv3_r50.pth
WORK_DIR=work_dirs/eval_pretrained_$(date +%Y%m%d_%H%M%S)
mkdir -p "${WORK_DIR}"

echo ">>> Eval pretrained Sparse4Dv3 R50 with 8 GPUs"
echo "    config = ${CONFIG}"
echo "    ckpt   = ${CKPT}"
echo "    output = ${WORK_DIR}"

bash tools/dist_test.sh \
    "${CONFIG}" \
    "${CKPT}" \
    8 \
    --eval bbox \
    --eval-options jsonfile_prefix="${WORK_DIR}/results" \
    2>&1 | tee "${WORK_DIR}/eval.log"

echo ""
echo "[DONE] Eval log: ${WORK_DIR}/eval.log"
echo "       metrics : ${WORK_DIR}/results_pts_bbox/metrics_summary.json"
