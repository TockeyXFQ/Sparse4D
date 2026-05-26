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

    与上游唯一的差异是把 `r = r.append(copy)` 换成 `r = pd.concat([r, copy])`,
    其余逻辑(frame / OId / HId 重映射)与上游 nuscenes-devkit v1.1.x 完全一致。
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
            next_frame_id = max(
                r.index.get_level_values(0).max() + 1,
                r.index.get_level_values(0).unique().shape[0],
            )
            if np.isnan(next_frame_id):
                next_frame_id = 0
            copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
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


_apply_nuscenes_tracking_pandas2_patch()
_apply_nuscenes_tracking_nan_unique_patch()
