"""nuscenes-devkit / motmetrics 与 pandas 2.x 的兼容性补丁。

nuscenes-devkit v1.1.x 的 `nuscenes/eval/tracking/mot.py:MOTAccumulatorCustom.merge_event_dataframes`
仍在使用 pandas 1.x 的 `DataFrame.append` —— 该 API 在 pandas 1.4 被废弃,
在 pandas 2.0 已被彻底移除,改为推荐使用 `pd.concat`。

容器内 pandas==2.3.1,所以 tracking 评估会在 20 epoch 第一次评估时直接抛
`AttributeError: 'DataFrame' object has no attribute 'append'`,导致整个分布式训练 job 退出。

本模块通过 monkey-patch 替换该方法,使 tracking 评估在 pandas>=2.0 环境下也能正常运行。
导入 `projects.mmdet3d_plugin` 时自动生效。
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


_apply_nuscenes_tracking_pandas2_patch()
