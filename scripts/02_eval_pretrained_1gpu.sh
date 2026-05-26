#!/bin/bash
# =============================================================================
# 单卡版: 用作者放出的 Sparse4Dv3 R50 ckpt 跑 detection + tracking 双评测
# 目的: 对齐评测 pipeline,在只有 1 张 H20 的开发机上验证整套流程
# 期望时间: ~3-4 小时 (单卡 H20, 6019 个 val sample,~2-3s/sample)
# 期望指标 (config 注释里官方数字, val split):
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

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PWD}:${PYTHONPATH:-}
export PORT=29513

# 不直接调 tools/test.py 走 MMDataParallel,因为 mmcv 1.7.2 的 MMDataParallel 在
# torch 2.1 上有已知兼容 bug(_get_stream 把 device id int 当成 torch.device 传)。
# 改用 dist_test.sh 单卡分布式模式(走 MMDistributedDataParallel,兼容)。

CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
CKPT=ckpt/sparse4dv3_r50.pth
WORK_DIR=work_dirs/eval_pretrained_1gpu_$(date +%Y%m%d_%H%M%S)
mkdir -p "${WORK_DIR}"

echo ">>> Eval pretrained Sparse4Dv3 R50 with 1 GPU (single-process DDP)"
echo "    config = ${CONFIG}"
echo "    ckpt   = ${CKPT}"
echo "    output = ${WORK_DIR}"

bash tools/dist_test.sh \
    "${CONFIG}" \
    "${CKPT}" \
    1 \
    --out "${WORK_DIR}/outputs.pkl" \
    --eval bbox \
    --eval-options jsonfile_prefix="${WORK_DIR}/results" \
    2>&1 | tee "${WORK_DIR}/eval.log"

echo ""
echo "[DONE] Eval log: ${WORK_DIR}/eval.log"
echo "       metrics : ${WORK_DIR}/results_pts_bbox/metrics_summary.json"
