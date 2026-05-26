#!/bin/bash
# =============================================================================
# 单卡 smoke-test 训练: 跑 200 iter 验证训练流程能跑通 + loss 正常下降
# 不用于复现论文数字(只有 1 卡,跑完 100 epoch 要 8 天)
# 期望时间: ~5-10 分钟
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "${SPARSE4D_ROOT}"

# ----- Python 环境选择 (3 种模式,优先级从高到低) -----
# 1. USE_GLOBAL_PYTHON=1  → 系统 /usr/local/bin/python3.11(镜像模式)
# 2. SPARSE4D_VENV=<path> → 指定 venv 路径(默认 /opt/sparse4d_env)
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
        echo "[WARN] No venv at ${SPARSE4D_VENV}, falling back to system python"
    fi
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=${PWD}:${PYTHONPATH:-}
export PORT=28652
export OMP_NUM_THREADS=8

CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
WORK_DIR=work_dirs/baseline_v3_r50_1gpu_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "${WORK_DIR}"

# smoke-test 的 cfg 覆盖:
# - runner.max_iters=200       只跑 200 iter (~5-10 min)
# - evaluation.interval=百万    不触发 val eval(快)
# - checkpoint_config.interval=百万   不存中间 ckpt(快)
# - data.workers_per_gpu=4     1 卡时 worker 不需要太多
echo ">>> Smoke-test training Sparse4Dv3 R50 on 1 GPU (200 iter)"
echo "    config   = ${CONFIG}"
echo "    work dir = ${WORK_DIR}"

bash tools/dist_train.sh \
    "${CONFIG}" \
    1 \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
        runner.max_iters=200 \
        evaluation.interval=1000000 \
        checkpoint_config.interval=1000000 \
        data.workers_per_gpu=4 \
    2>&1 | tee "${WORK_DIR}/train.log"

echo ""
echo "[DONE] Smoke-test finished."
echo "       log: ${WORK_DIR}/train.log"
echo "  Healthy: loss_cls + loss_box 在前 100 iter 内应从 ~10 降到 ~3-5"
