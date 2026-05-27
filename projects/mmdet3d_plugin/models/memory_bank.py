"""A2 - MemoryBank: long-horizon temporal cache for Sparse4D.

Phase 1 A2:替代 ``InstanceBank`` 的小 ring-buffer (num_temp_instances=8) 为
大容量长程记忆池 (num_memory_instances=512,64× 容量),目标是长遮挡 case 的
IDS 显著下降。设计上**完全保持 head 接口透明**:对外暴露的 ``get()`` 仍然返回
top-``num_temp_instances`` 个 instance 给 head 当 temporal input,只是这些
instance 是从大 memory pool 里筛出来的(经过更长的 confidence_decay)。

Why it helps (机制):
  nuScenes 2 Hz 采样下,默认 InstanceBank 的 8 帧 ring buffer 只能记住 4s
  前的 instance。一个被遮挡 ≥4 秒的车再出现,query feature 已经被冲掉,
  Sparse4D 只能给新 ID,IDS+1。512 容量的 memory pool 能记住几十秒前出现过
  的 instance,长遮挡 reappear 直接用 cached feature 续上同一 ID。

实现要点(跟 InstanceBank 保持向后兼容):
  - 继承 ``InstanceBank``,仅 override ``cache()`` 和 ``reset()``
  - ``self.memory_pool_*`` 是真正的长程 cache (num_memory_instances 容量)
  - ``self.cached_*`` 仍然只保留 num_temp_instances 个 (head 直接读这个)
  - ``get()`` / ``update()`` / ``get_instance_id()`` 完全继承,无需重写
  - 配置上:``num_temp_instances`` 保持原值(比如 600),只新增
    ``num_memory_instances=512`` 控制记忆池大小

Risks / known-good defaults:
  - ``num_memory_instances`` 太大也会引入 false continuation(过期 track 错误
    延续到新物体),chenxi 工程默认 512,做了 256/512/1024 三档实验
  - 长程 memory 一旦 enable,confidence_decay 应该调大(默认 0.6 → 0.8?),
    避免老 instance confidence 衰减过快被快速踢出 pool
"""
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from .instance_bank import InstanceBank, topk

__all__ = ["MemoryBank"]


@PLUGIN_LAYERS.register_module()
class MemoryBank(InstanceBank):
    """Long-horizon memory bank (Phase 1 A2).

    Args:
        num_memory_instances: long-horizon memory pool 大小(默认 512,
            chenxi/onemodel 实战值)。当 num_memory_instances <= num_temp_instances
            时退化成普通 InstanceBank 行为。
        memory_confidence_decay: memory pool 内的 confidence decay 因子,
            默认 0.85(比 baseline 的 0.6 更慢衰减,因为容量大需要更长生命周期)。
        其他参数同 ``InstanceBank``。
    """

    def __init__(
        self,
        num_memory_instances: int = 512,
        memory_confidence_decay: float = 0.85,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_memory_instances = max(
            int(num_memory_instances), int(self.num_temp_instances)
        )
        self.memory_confidence_decay = float(memory_confidence_decay)

    def reset(self):
        super().reset()
        # memory pool 在切 scene 时清空(基类 reset 不知道这些字段)
        self.memory_feature = None
        self.memory_anchor = None
        self.memory_confidence = None

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        """Update both the long-horizon memory pool AND the small temp cache.

        Order of operations(关键):
            1. 更新 memory pool (容量 num_memory_instances,confidence_decay=
               memory_confidence_decay 较慢衰减)
            2. 从 memory pool 里 topk num_temp_instances 个写到 self.cached_*
               (这是 head.get() 实际读的)
            3. self.confidence / self.temp_confidence 跟 InstanceBank 保持兼容
               (供 update_instance_id 用)
        """
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        # 当前帧 instance 的 confidence (B, A)
        cur_confidence = confidence.max(dim=-1).values.sigmoid()
        bs, num_cur = cur_confidence.shape

        # ---- 1. 跟 memory pool 合并 ----
        if self.memory_feature is None or self.memory_feature.shape[0] != bs:
            # 第一帧 / batch size 变了 -> 重新初始化 memory pool
            merged_feature = instance_feature
            merged_anchor = anchor
            merged_confidence = cur_confidence
        else:
            # memory pool 的旧 confidence 先 decay (越老越低)
            decayed_memory_confidence = (
                self.memory_confidence * self.memory_confidence_decay
            )
            # 当前帧 + 老 memory 拼起来 (B, A + num_memory, ...)
            merged_feature = torch.cat(
                [instance_feature, self.memory_feature], dim=1
            )
            merged_anchor = torch.cat([anchor, self.memory_anchor], dim=1)
            merged_confidence = torch.cat(
                [cur_confidence, decayed_memory_confidence], dim=1
            )

        # ---- 2. topk -> memory pool 新状态 (容量 num_memory_instances) ----
        cap = min(self.num_memory_instances, merged_feature.shape[1])
        (
            self.memory_confidence,
            (self.memory_feature, self.memory_anchor),
        ) = topk(merged_confidence, cap, merged_feature, merged_anchor)

        # ---- 3. 从 memory pool 里 topk num_temp_instances 给 head 当 temp ----
        # 这一步保证 head.forward 拿到的 self.cached_* 形状跟 baseline 一致,
        # 只是内容来自更大的 memory pool 而非小 ring buffer
        num_temp = min(self.num_temp_instances, cap)
        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(self.memory_confidence, num_temp, self.memory_feature, self.memory_anchor)

        # temp_confidence 是 InstanceBank.update_instance_id() 用来给 instance_id
        # 排序的字段;baseline 是把 (cur + decayed_old) 合并后的 conf 写到这里,
        # 形状是 (B, num_anchor)。MemoryBank 里我们保持兼容:用 merged_confidence
        # 的前 num_cur 个 (对应当前帧 instance 顺序),这样 update_instance_id 接
        # 着用就行。
        self.temp_confidence = merged_confidence[:, :num_cur]
