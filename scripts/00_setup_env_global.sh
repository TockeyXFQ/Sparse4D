#!/bin/bash
# =============================================================================
# Sparse4D-V3 全局环境配置 (适用任务平台镜像模式)
# =============================================================================
# 用法场景:
#   - 你的开发机已经有公司提供的 AI 训练镜像(torch/mmcv/mmdet 全套)
#   - 训练时挂载镜像 + 数据卷 + 提交命令,无法 SSH 进去 source venv
#   - 所以把额外依赖装到系统 /usr/local/lib/python3.11/site-packages/
#   - 装完后保存镜像,8 卡训练时直接用
#
# 这个脚本相对于 00_setup_env.sh 的差别:
#   - 不创建 .venv,直接用系统 /usr/local/bin/python3.11
#   - 只补装系统缺的包(目前只有 motmetrics)
#   - 用 setup.py install(而非 develop)把 deformable_aggregation 装到全局
#   - 跑 _apply_compat_patches.sh 给系统 mmcv / motmetrics 打补丁
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
PYTHON_BIN=/usr/local/bin/python3.11
PIP_BIN="${PYTHON_BIN} -m pip"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=${CUDA_HOME}/bin:${PATH:-}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

echo ">>> Sparse4D-V3 root  : ${SPARSE4D_ROOT}"
echo ">>> Python interpreter: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"
echo ">>> CUDA_HOME         : ${CUDA_HOME}"
echo ""

# --------------------------------------------------------------------------- #
# 1. 检查系统已装的关键包(目标: torch ≥2.1+cu121, mmcv 1.7.x, mmdet 2.28.x)
# --------------------------------------------------------------------------- #
echo "========== STEP 1: 检查系统已装包 =========="
${PYTHON_BIN} - <<'PY'
import importlib, sys
need = {
    "torch":   "2.1",   # must be cu121 to match system CUDA
    "mmcv":    "1.7",   # 1.7.x with full ops
    "mmdet":   "2.28",  # mmdet 2.x series
    "numpy":   "1.",
}
missing = []
for mod, prefix in need.items():
    try:
        m = importlib.import_module(mod)
        ver = m.__version__
        ok = ver.startswith(prefix)
        flag = "OK " if ok else "WARN"
        print(f"  [{flag}] {mod:<10s} {ver}")
        if not ok:
            print(f"          expected prefix: {prefix}")
    except ImportError:
        missing.append(mod)
        print(f"  [MISS] {mod}: not installed")
if missing:
    print(f"\n[FAIL] missing packages: {missing}")
    sys.exit(1)
PY
echo ""

# --------------------------------------------------------------------------- #
# 2. 补装系统缺的包(motmetrics + sklearn 给 anchor_generator)
# --------------------------------------------------------------------------- #
echo "========== STEP 2: 补装缺失依赖 =========="
${PIP_BIN} install --no-cache-dir motmetrics==1.1.3 2>&1 | tail -3
# scikit-learn 99% 装好了,但保险起见装一次(--upgrade-strategy only-if-needed 默认不重装已有)
${PIP_BIN} install --no-cache-dir scikit-learn 2>&1 | tail -3
echo ""

# --------------------------------------------------------------------------- #
# 3. 应用兼容性 patch(mmcv distributed/_functions + motmetrics metrics)
# --------------------------------------------------------------------------- #
echo "========== STEP 3: 应用 Py3.10/3.11 + torch 2.x 兼容补丁 =========="
PYTHON_BIN="${PYTHON_BIN}" bash "${SPARSE4D_ROOT}/scripts/_apply_compat_patches.sh"
echo ""

# --------------------------------------------------------------------------- #
# 4. 编译 + 安装 deformable_aggregation CUDA op 到全局
# --------------------------------------------------------------------------- #
echo "========== STEP 4: 编译 deformable_aggregation CUDA op (sm_90) =========="
pushd "${SPARSE4D_ROOT}/projects/mmdet3d_plugin/ops" > /dev/null
# 清 build 临时目录(不删源码目录的 .so,避免破坏其他 venv 的 develop 安装)
rm -rf build *.egg-info
# 装到 site-packages 而非 develop(镜像迁移友好)
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0+PTX" \
    ${PYTHON_BIN} setup.py install 2>&1 | tail -10
# 把编译产物 .so 同时拷一份到源码目录,这样 venv 走 develop 模式时仍能找到 op
COMPILED_SO=$(find /usr/local/lib/python3.11/site-packages -maxdepth 4 -name "deformable_aggregation_ext*.so" 2>/dev/null | head -1)
if [ -n "${COMPILED_SO}" ]; then
    cp -f "${COMPILED_SO}" .
    echo "[OK] symlink-back .so to source dir for venv develop mode"
fi
popd > /dev/null
echo ""

# --------------------------------------------------------------------------- #
# 5. 最终验证 - 用系统 python 跑一次完整 import
# --------------------------------------------------------------------------- #
echo "========== STEP 5: 验证 =========="
cd "${SPARSE4D_ROOT}"
PYTHONPATH=${PWD}:${PYTHONPATH:-} ${PYTHON_BIN} - <<'PY'
import sys, torch, mmcv, mmdet, numpy, motmetrics
from projects.mmdet3d_plugin.ops.deformable_aggregation import DeformableAggregationFunction
from mmcv.parallel import MMDistributedDataParallel
from nuscenes.eval.tracking.evaluate import TrackingEval
from motmetrics.metrics import MetricsHost
print("=" * 60)
print(f"Python       : {sys.version.split()[0]}")
print(f"Interpreter  : {sys.executable}")
print(f"torch        : {torch.__version__} (cuda={torch.version.cuda})")
print(f"GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU cap      : sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}" if torch.cuda.is_available() else "")
print(f"mmcv         : {mmcv.__version__}")
print(f"mmdet        : {mmdet.__version__}")
print(f"numpy        : {numpy.__version__}")
print(f"motmetrics   : {motmetrics.__version__}")
print(f"deformable_aggregation : OK (.so installed)")
print(f"MMDistributedDataParallel : OK (patched)")
print(f"TrackingEval / MetricsHost : OK (motmetrics patched)")
print("=" * 60)
PY

echo ""
echo "[DONE] Global environment ready. Save image now if desired."
echo "       For training, no 'source venv' needed; system python is good to go."
