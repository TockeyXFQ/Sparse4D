#!/bin/bash
# =============================================================================
# 幂等地把 mmcv 1.7.x + motmetrics 1.1.x 上的 Python 3.10/3.11 + torch 2.x
# 兼容性补丁打上(可重复执行,已 patch 过的不会重复改)。
#
# 涉及补丁:
#   1. mmcv/parallel/distributed.py  — `_use_replicated_tensor_module` 缺属性
#                                       (公司基础镜像已 patch,但 venv 没有)
#   2. mmcv/parallel/_functions.py   — `_get_stream(int)` 在 torch 2.x 报错
#                                       (传 int 给 torch 2.x 的 `_get_stream` 会
#                                        抛 'int' object has no attribute 'type')
#   3. motmetrics/metrics.py         — `from collections import Iterable`
#                                       Py3.10+ 已 deprecated → collections.abc
#   4. motmetrics/metrics.py         — `inspect.getargspec` Py3.11 已移除
#                                       → getfullargspec
#
# 用法:
#   bash scripts/_apply_compat_patches.sh                 # patch system & venv
#   PYTHON_BIN=/path/to/python bash scripts/_apply_compat_patches.sh
# =============================================================================
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"
SITE_PACKAGES=$("${PYTHON_BIN}" -c "import site, sys; [print(p) for p in site.getsitepackages() if 'site-packages' in p][:1]" | head -1)

echo ">>> Patching site-packages: ${SITE_PACKAGES}"

# --------------------------------------------------------------------------- #
# Patch 1: mmcv/parallel/distributed.py _use_replicated_tensor_module
# --------------------------------------------------------------------------- #
MMCV_DIST=${SITE_PACKAGES}/mmcv/parallel/distributed.py
if [ -f "${MMCV_DIST}" ]; then
    if grep -q "_use_replicated_tensor_module else self.module" "${MMCV_DIST}" 2>/dev/null \
        && ! grep -q "Compat patch for torch >= 2.0" "${MMCV_DIST}" 2>/dev/null; then
        # 用 python 替换,避免多行 sed escape 麻烦
        "${PYTHON_BIN}" - <<PY
import re, pathlib
p = pathlib.Path("${MMCV_DIST}")
src = p.read_text()
old = (
    "        module_to_run = self._replicated_tensor_module if \\\\\n"
    "            self._use_replicated_tensor_module else self.module"
)
new = (
    "        # Compat patch for torch >= 2.0:\n"
    "        # _use_replicated_tensor_module / _replicated_tensor_module were\n"
    "        # removed in torch 2.0 DDP and never set, so attribute access raises.\n"
    "        if getattr(self, \"_use_replicated_tensor_module\", False):\n"
    "            module_to_run = self._replicated_tensor_module\n"
    "        else:\n"
    "            module_to_run = self.module"
)
if old in src:
    p.write_text(src.replace(old, new))
    print("[OK] patched mmcv/parallel/distributed.py")
else:
    print("[SKIP] mmcv/parallel/distributed.py — pattern not found (already patched or different style)")
PY
    else
        echo "[SKIP] mmcv/parallel/distributed.py — already patched"
    fi
else
    echo "[MISS] ${MMCV_DIST} — mmcv not installed for this python"
fi

# --------------------------------------------------------------------------- #
# Patch 2: mmcv/parallel/_functions.py — _get_stream(int) → _get_stream(device)
# --------------------------------------------------------------------------- #
MMCV_FUNC=${SITE_PACKAGES}/mmcv/parallel/_functions.py
if [ -f "${MMCV_FUNC}" ]; then
    if grep -qE "streams = \[_get_stream\(device\) for device in (target_gpus|temp_target_gpus)\]" "${MMCV_FUNC}" 2>/dev/null \
        && ! grep -q "Compat patch for torch >= 2.0" "${MMCV_FUNC}" 2>/dev/null; then
        "${PYTHON_BIN}" - <<PY
import pathlib, re
p = pathlib.Path("${MMCV_FUNC}")
src = p.read_text()
pattern = re.compile(
    r"^(            )streams = \[_get_stream\(device\) for device in (\w+)\]\n",
    re.MULTILINE,
)
def repl(m):
    indent = m.group(1)
    varname = m.group(2)
    return (
        f"{indent}# Compat patch for torch >= 2.0: _get_stream expects torch.device,\n"
        f"{indent}# but mmcv 1.x passes raw int gpu id.\n"
        f"{indent}streams = [\n"
        f"{indent}    _get_stream(\n"
        f"{indent}        torch.device('cuda', d) if isinstance(d, int) else d\n"
        f"{indent}    )\n"
        f"{indent}    for d in {varname}\n"
        f"{indent}]\n"
    )
new_src, n = pattern.subn(repl, src)
if n > 0:
    p.write_text(new_src)
    print(f"[OK] patched mmcv/parallel/_functions.py (n={n})")
else:
    print("[SKIP] mmcv/parallel/_functions.py — pattern not found (already patched or different style)")
PY
    else
        echo "[SKIP] mmcv/parallel/_functions.py — already patched"
    fi
else
    echo "[MISS] ${MMCV_FUNC} — mmcv not installed for this python"
fi

# --------------------------------------------------------------------------- #
# Patch 3 + 4: motmetrics/metrics.py
# --------------------------------------------------------------------------- #
MM_METRICS=${SITE_PACKAGES}/motmetrics/metrics.py
if [ -f "${MM_METRICS}" ]; then
    if grep -q "from collections import OrderedDict, Iterable" "${MM_METRICS}"; then
        sed -i \
            -e 's|from collections import OrderedDict, Iterable|from collections import OrderedDict\nfrom collections.abc import Iterable  # Py3.10+ moved Iterable to collections.abc|' \
            "${MM_METRICS}"
        echo "[OK] patched motmetrics/metrics.py — collections.Iterable"
    else
        echo "[SKIP] motmetrics/metrics.py — collections.Iterable already patched"
    fi

    if grep -q "inspect.getargspec(fnc)" "${MM_METRICS}"; then
        sed -i \
            -e 's|inspect\.getargspec(fnc)|inspect.getfullargspec(fnc)|' \
            "${MM_METRICS}"
        echo "[OK] patched motmetrics/metrics.py — inspect.getargspec"
    else
        echo "[SKIP] motmetrics/metrics.py — inspect.getargspec already patched"
    fi
else
    echo "[MISS] ${MM_METRICS} — motmetrics not installed for this python"
fi

# --------------------------------------------------------------------------- #
# Final validation: try to import everything Sparse4D evaluation needs
# --------------------------------------------------------------------------- #
"${PYTHON_BIN}" - <<'PY'
import sys
errors = []
try:
    import torch, mmcv, mmdet, numpy
    print(f"torch={torch.__version__}, mmcv={mmcv.__version__}, mmdet={mmdet.__version__}, numpy={numpy.__version__}")
except Exception as e:
    errors.append(("base", e))

# DDP forward path
try:
    from mmcv.parallel import MMDistributedDataParallel  # noqa
    print("MMDistributedDataParallel import: OK")
except Exception as e:
    errors.append(("mmcv DDP", e))

# motmetrics (tracking eval)
try:
    import motmetrics  # noqa
    from motmetrics.metrics import MetricsHost  # triggers all the patched imports
    print(f"motmetrics={motmetrics.__version__}: OK")
except Exception as e:
    errors.append(("motmetrics", e))

# nuscenes tracking eval entry
try:
    from nuscenes.eval.tracking.evaluate import TrackingEval  # noqa
    print("nuscenes TrackingEval import: OK")
except Exception as e:
    errors.append(("TrackingEval", e))

if errors:
    print("\n[FAIL] Some imports failed:")
    for name, e in errors:
        print(f"  {name}: {type(e).__name__}: {e}")
    sys.exit(1)
print("\n[OK] All compat-critical imports succeeded.")
PY

echo ""
echo "[DONE] compat patches applied."
