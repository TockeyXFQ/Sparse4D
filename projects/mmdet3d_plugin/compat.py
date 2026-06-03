"""nuscenes-devkit 与 pandas 2.x / numpy 1.x 的兼容性补丁集合。

容器内 pandas==2.3.1, numpy==1.26.4。nuscenes-devkit v1.1.x 的 tracking 评估代码
有两处不兼容,会让训练在第一次 tracking eval(20 epoch)时打挂分布式 job。本模块通过
monkey-patch 修复,导入 `projects.mmdet3d_plugin` 时自动生效。

Patch 1: pandas 2.x 兼容
    `nuscenes/eval/tracking/mot.py:MOTAccumulatorCustom.merge_event_dataframes` 使用了
    pandas 1.x 的 `DataFrame.append`(2.0 已移除),替换为 `pd.concat`。

Patch 2: numpy 1.x 的 np.unique 对 NaN 的处理
    `nuscenes/eval/tracking/algo.py:161` 有这样一个 assert:
        assert unachieved + duplicate + len(thresh_metrics) == num_thresholds
    其中 `duplicate = len(thresholds) - len(np.unique(thresholds))`。
    numpy 1.x 的 `np.unique` 会把所有 NaN 合并成 1 个,但 numpy 2.0+ 保留全部。
    nuscenes-devkit 的逻辑假定了 numpy 2.0 行为,所以在 numpy 1.x 下只要 thresholds 含 NaN
    必然 assert 失败。我们在 `algo` 模块内把 `np.unique` 替换成保留所有 NaN 的版本。

Patch 3: gradient checkpointing 默认走非重入(use_reentrant=False)
    mmdet 的 `ResNet(with_cp=True)` 内部用 `torch.utils.checkpoint.checkpoint(fn, x)`,
    未显式传 `use_reentrant`,torch<2.4 默认 reentrant(重入)模式。重入式 checkpoint 会在
    backward 阶段重跑 forward,把 autograd hook 再触发一遍。当 DDP 配置
    `find_unused_parameters=True` 时(F4 masked-modal 必须开),这会让同一个参数被 DDP
    标记 ready 两次,第一个 backward 就崩:
        RuntimeError: Expected to mark a variable ready only once
        Parameter at index 157 (img_backbone.layer4.2.bn3.weight) marked ready twice
    static_graph 不可用(masking 每步参与的参数集会变)。torch 官方推荐的解法是把检查点
    切成非重入(use_reentrant=False),它专门兼容 find_unused_parameters=True,且同样省显存。
    本 patch 在调用方未显式指定 use_reentrant 时,把默认值改成 False。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from itertools import count

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _patched_merge_event_dataframes(
    dfs,
    update_frame_indices: bool = True,
    update_oids: bool = True,
    update_hids: bool = True,
    return_mappings: bool = False,
):
    """Replacement for `MOTAccumulatorCustom.merge_event_dataframes` (pandas 2.x safe).

    与上游 nuscenes-devkit v1.1.x 的差异:
      1. `r.append(copy)` → `pd.concat(...)` (pandas 2.0 移除了 DataFrame.append)
      2. `copy.index.map(lambda)` 后显式 `MultiIndex.from_tuples` 重建 MultiIndex —
         pandas 2.x 下 MultiIndex.map 不再自动保持 MultiIndex,会退化成 Index of tuples,
         导致下一轮 `get_level_values(0).max() + 1` 抛 `TypeError: tuple + int`
      3. 跳过空 DataFrame 的 concat 以避免 pandas 2.x 的 FutureWarning
    其余逻辑(frame / OId / HId 重映射)与上游完全一致。
    """
    from nuscenes.eval.tracking.mot import MOTAccumulatorCustom

    mapping_infos = []
    new_oid = count()
    new_hid = count()

    r = MOTAccumulatorCustom.new_event_dataframe()
    for df in dfs:
        if isinstance(df, MOTAccumulatorCustom):
            df = df.events

        copy = df.copy()
        infos = {}

        if update_frame_indices:
            # r 为空时 .max() 返回 NaN,后续 NaN+1 仍为 NaN,被 np.isnan 兜底为 0
            level0 = r.index.get_level_values(0)
            next_frame_id = max(
                (level0.max() + 1) if len(level0) > 0 else 0,
                level0.unique().shape[0],
            )
            if isinstance(next_frame_id, float) and np.isnan(next_frame_id):
                next_frame_id = 0
            if len(copy.index) > 0:
                names = copy.index.names
                shifted = [(x[0] + next_frame_id, x[1]) for x in copy.index]
                copy.index = pd.MultiIndex.from_tuples(shifted, names=names)
            infos["frame_offset"] = next_frame_id

        if update_oids:
            oid_map = OrderedDict(
                (oid, str(next(new_oid))) for oid in copy["OId"].dropna().unique()
            )
            copy["OId"] = copy["OId"].map(lambda x: oid_map[x], na_action="ignore")
            infos["oid_map"] = oid_map

        if update_hids:
            hid_map = OrderedDict(
                (hid, str(next(new_hid))) for hid in copy["HId"].dropna().unique()
            )
            copy["HId"] = copy["HId"].map(lambda x: hid_map[x], na_action="ignore")
            infos["hid_map"] = hid_map

        if len(r) == 0:
            r = copy
        elif len(copy) > 0:
            r = pd.concat([r, copy])
        mapping_infos.append(infos)

    if return_mappings:
        return r, mapping_infos
    return r


def _apply_nuscenes_tracking_pandas2_patch() -> None:
    try:
        from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
    except ImportError:
        logger.debug("nuscenes-devkit not installed; skip tracking eval patch.")
        return

    if getattr(MOTAccumulatorCustom, "_sparse4d_pandas2_patched", False):
        return

    MOTAccumulatorCustom.merge_event_dataframes = staticmethod(
        _patched_merge_event_dataframes
    )
    MOTAccumulatorCustom._sparse4d_pandas2_patched = True
    logger.info(
        "Patched nuscenes.eval.tracking.mot.MOTAccumulatorCustom.merge_event_dataframes "
        "for pandas>=2.0 compatibility."
    )


def _unique_numpy2_style(ar, *args, **kwargs):
    """np.unique 的 numpy>=2.0 兼容版本:对浮点数组保留全部 NaN(而非合并为 1 个)。

    `args/kwargs` 全部转发给原 `np.unique`。仅当数组中含 NaN 时走我们的回退路径,
    其他情况完全等价于上游 numpy 实现。
    """
    arr = np.asarray(ar)
    if arr.dtype.kind != "f" or not np.any(np.isnan(arr)):
        return _orig_np_unique(ar, *args, **kwargs)

    nan_count = int(np.sum(np.isnan(arr)))
    non_nan = arr[~np.isnan(arr)]
    unique_non_nan = _orig_np_unique(non_nan, *args, **kwargs)
    return np.concatenate([np.asarray(unique_non_nan), np.full(nan_count, np.nan)])


_orig_np_unique = np.unique


class _NumpyUniqueShim:
    """一个仅重写 `unique` 的 numpy 代理,其余属性透传到真 numpy 模块。

    用法:`module.np = _NumpyUniqueShim(np)` 之后,该模块内的 `np.unique` 走我们的实现,
    `np.<其他>` 仍是上游 numpy。"""

    __slots__ = ("_np",)

    def __init__(self, np_module):
        object.__setattr__(self, "_np", np_module)

    def __getattr__(self, name):
        if name == "unique":
            return _unique_numpy2_style
        return getattr(self._np, name)


def _apply_nuscenes_tracking_nan_unique_patch() -> None:
    try:
        from nuscenes.eval.tracking import algo as _algo_mod
    except ImportError:
        logger.debug("nuscenes-devkit not installed; skip tracking algo patch.")
        return

    if getattr(_algo_mod, "_sparse4d_nan_unique_patched", False):
        return

    _algo_mod.np = _NumpyUniqueShim(np)
    _algo_mod._sparse4d_nan_unique_patched = True
    logger.info(
        "Patched nuscenes.eval.tracking.algo: np.unique now retains all NaN values "
        "(numpy>=2.0 behavior), fixing the assert at tracking/algo.py:161 under numpy<2.0."
    )


def _apply_nonreentrant_checkpoint_patch() -> None:
    """把 torch.utils.checkpoint.checkpoint 默认改成非重入(use_reentrant=False)。

    仅当调用方未显式传 `use_reentrant` 时才注入 False;显式传入的(无论 True/False)
    一律尊重原意。幂等:重复 import 不会二次包装。
    """
    try:
        import torch.utils.checkpoint as _cp
    except ImportError:
        logger.debug("torch not installed; skip checkpoint patch.")
        return

    if getattr(_cp, "_sparse4d_nonreentrant_patched", False):
        return

    _orig_checkpoint = _cp.checkpoint

    def _patched_checkpoint(*args, **kwargs):
        if "use_reentrant" not in kwargs:
            kwargs["use_reentrant"] = False
        return _orig_checkpoint(*args, **kwargs)

    _cp.checkpoint = _patched_checkpoint
    _cp._sparse4d_nonreentrant_patched = True
    logger.info(
        "Patched torch.utils.checkpoint.checkpoint: use_reentrant now defaults to "
        "False (non-reentrant), compatible with DDP find_unused_parameters=True."
    )


_apply_nuscenes_tracking_pandas2_patch()
_apply_nuscenes_tracking_nan_unique_patch()
_apply_nonreentrant_checkpoint_patch()
