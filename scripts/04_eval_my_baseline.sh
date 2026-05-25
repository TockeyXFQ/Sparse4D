#!/bin/bash
# =============================================================================
# 评测自己训出的 baseline ckpt(detection + tracking 双指标)
# 用法: bash scripts/04_eval_my_baseline.sh [ckpt_path]
# 默认 ckpt: work_dirs/baseline_v3_r50_repro/latest.pth
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "${SPARSE4D_ROOT}"

# shellcheck disable=SC1091
source "${SPARSE4D_ROOT}/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=${PWD}:${PYTHONPATH:-}
export PORT=29513

CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
CKPT=${1:-work_dirs/baseline_v3_r50_repro/latest.pth}
WORK_DIR=work_dirs/eval_my_baseline_$(date +%Y%m%d_%H%M%S)
mkdir -p "${WORK_DIR}"

if [ ! -f "${CKPT}" ]; then
    echo "[FAIL] checkpoint not found: ${CKPT}"
    exit 1
fi

echo ">>> Eval my trained baseline with 8 GPUs"
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
