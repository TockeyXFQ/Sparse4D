#!/bin/bash
# =============================================================================
# Sparse4D-V3 一键环境配置 (适配 Python 3.11 + 8卡 H20 + 系统 CUDA 12.1)
# =============================================================================
# 关键选型决策:
# 1) Python 3.11(/usr/local/bin):系统自带的 Python 3.10 缺 Python.h / lib2to3
#    / distutils(Debian minimal 安装),无 sudo 没法补;3.11 是完整安装
# 2) torch 2.1.2+cu121:严格匹配系统 CUDA 12.1(torch.cpp_extension 强校验)
# 3) mmcv-full 1.7.2:OpenMMLab 有 cu121/torch2.1/cp311 预编译 wheel
# 4) mmdet 2.28.2:与 mmcv-full 1.x 配套,Sparse4D 代码直接可用
# 5) deformable_aggregation CUDA op 显式带 sm_90 PTX 重新编译(H20 是 Hopper)
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
VENV_PATH=${SPARSE4D_ROOT}/.venv
PYTHON_BIN=/usr/local/bin/python3.11

# 系统 CUDA toolkit 路径(系统 driver 12.8,toolkit 12.1)
# 编译 deformable_aggregation CUDA op 时需要 nvcc
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=${CUDA_HOME}/bin:${PATH:-}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

echo ">>> Sparse4D-V3 root: ${SPARSE4D_ROOT}"
echo ">>> Will create venv at: ${VENV_PATH}"
echo ">>> Python interpreter: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"
echo ">>> CUDA_HOME = ${CUDA_HOME}"
echo ">>> nvcc      = $(which nvcc) ($(nvcc --version | grep release | head -1))"

# --------------------------------------------------------------------------- #
# 1. 创建 Python 3.11 虚拟环境(Python 3.11 完整,有 ensurepip)
# --------------------------------------------------------------------------- #
if [ ! -d "${VENV_PATH}" ]; then
    ${PYTHON_BIN} -m venv "${VENV_PATH}"
    echo "[OK] venv created (Python 3.11)"
else
    echo "[SKIP] venv already exists at ${VENV_PATH}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

# --------------------------------------------------------------------------- #
# 2. 升级 pip 工具链
# --------------------------------------------------------------------------- #
pip install --upgrade pip==23.3.2 setuptools==68.2.2 wheel==0.41.3

# --------------------------------------------------------------------------- #
# 3. 安装 torch 2.1.2 + cu121(匹配系统 CUDA 12.1,原生支持 H20 sm_90)
#    注: requirement.txt 标的是 torch 1.13+cu117,但系统 CUDA 12.1,torch
#    cpp_extension 强制 major 版本对齐才能编译 CUDA op,因此必须改用 cu121。
# --------------------------------------------------------------------------- #
pip install \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# --------------------------------------------------------------------------- #
# 4. 安装 mmcv-full 1.7.2(OpenMMLab 预编译,cu121/torch2.1/py310)
# --------------------------------------------------------------------------- #
pip install mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# --------------------------------------------------------------------------- #
# 5. 安装其他依赖(严格对齐 requirement.txt + 补 sklearn for anchor_generator)
# --------------------------------------------------------------------------- #
pip install \
    mmdet==2.28.2 \
    numpy==1.23.5 \
    urllib3==1.26.16 \
    pyquaternion==0.9.9 \
    nuscenes-devkit==1.1.10 \
    yapf==0.33.0 \
    tensorboard==2.14.0 \
    motmetrics==1.1.3 \
    pandas==1.5.3 \
    scikit-learn==1.3.2 \
    ipython

# --------------------------------------------------------------------------- #
# 6. 编译 deformable_aggregation CUDA op(包含 sm_90 PTX)
# --------------------------------------------------------------------------- #
pushd "${SPARSE4D_ROOT}/projects/mmdet3d_plugin/ops" > /dev/null
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX" \
    python setup.py develop 2>&1 | tail -20
popd > /dev/null

# --------------------------------------------------------------------------- #
# 7. 验收: 打印关键版本 + import deformable_aggregation
# --------------------------------------------------------------------------- #
python - <<'PY'
import torch, mmcv, mmdet, numpy
from projects.mmdet3d_plugin.ops.deformable_aggregation import DeformableAggregationFunction
print("=" * 60)
print(f"Python       : {__import__('sys').version.split()[0]}")
print(f"torch        : {torch.__version__}")
print(f"torch cuda   : {torch.version.cuda}")
print(f"cuda visible : {torch.cuda.device_count()} GPU(s)")
print(f"GPU name     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"compute cap  : {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'}")
print(f"mmcv         : {mmcv.__version__}")
print(f"mmdet        : {mmdet.__version__}")
print(f"numpy        : {numpy.__version__}")
print(f"deformable_aggregation: OK")
print("=" * 60)
PY

echo ""
echo "[DONE] Sparse4D-V3 environment ready."
echo "   Activate with:  source ${VENV_PATH}/bin/activate"
