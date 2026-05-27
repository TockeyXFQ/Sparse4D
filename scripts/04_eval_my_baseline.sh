#!/bin/bash
# =============================================================================
# 评测自己训出的 baseline ckpt(detection + tracking 双指标)
#
# 用法:
#   bash scripts/04_eval_my_baseline.sh [ckpt_path]                # 默认 8 卡
#   GPUS=1 bash scripts/04_eval_my_baseline.sh [ckpt_path]         # 1 卡(开发机)
#   USE_GLOBAL_PYTHON=1 GPUS=1 bash scripts/04_eval_my_baseline.sh # 镜像模式
#
# 默认 ckpt: work_dirs/baseline_v3_r50_repro/latest.pth
# 时长:8 卡 ~30 min;1 卡 ~45 min(detection) + 30 min(tracking) ≈ 1.25 h
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

# GPUS 数量(默认 8 卡训练机;开发机用 GPUS=1 override)
GPUS=${GPUS:-8}
case "${GPUS}" in
    1) export CUDA_VISIBLE_DEVICES=0 ;;
    8) export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    *) echo "[FAIL] GPUS=${GPUS} not supported (only 1 or 8)"; exit 1 ;;
esac
export PYTHONPATH=${PWD}:${PYTHONPATH:-}
export PORT=${PORT:-29513}

CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
CKPT=${1:-work_dirs/baseline_v3_r50_repro/latest.pth}
WORK_DIR=work_dirs/eval_my_baseline_$(date +%Y%m%d_%H%M%S)
mkdir -p "${WORK_DIR}"

if [ ! -f "${CKPT}" ]; then
    echo "[FAIL] checkpoint not found: ${CKPT}"
    exit 1
fi

echo ">>> Eval my trained baseline with ${GPUS} GPU(s)"
echo "    config = ${CONFIG}"
echo "    ckpt   = ${CKPT}"
echo "    output = ${WORK_DIR}"

# tracking_test=True 已在 config 里设置,evaluate() 会同时跑 detection + tracking
bash tools/dist_test.sh \
    "${CONFIG}" \
    "${CKPT}" \
    "${GPUS}" \
    --eval bbox \
    --eval-options jsonfile_prefix="${WORK_DIR}/results" \
    2>&1 | tee "${WORK_DIR}/eval.log"

echo ""
echo "[DONE] Eval log: ${WORK_DIR}/eval.log"
echo "       metrics : ${WORK_DIR}/results_pts_bbox/metrics_summary.json"
