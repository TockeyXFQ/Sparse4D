"""A4 - PF-Track style spatial-temporal reasoning for Sparse4D (simplified).

Phase 1 A4:基于 PF-Track (CVPR 2023, TRI-ML/PF-Track) 的核心 Future Reasoning
机制,简化适配 Sparse4D 接口。

PF-Track 论文有两个 sub-module:
    Past Reasoning  : history 帧 query → cross-attention refine 当前帧 query
    Future Reasoning: history N 帧 → 预测未来 K 帧 motion offset

Sparse4D-V3 原版的 ``temp_gnn`` op 已经实现了 Past Reasoning 的等价功能(query
attend to temp_instance_feature),所以这里**只实现 Future Reasoning** 这个增量
部分,作为 ``InstanceBank`` 子类 ``PFTrackInstanceBank``。

Why it helps (机制):
    Sparse4D 默认的 temporal cache 只能"复用过去看到过的位置"。一个被遮挡 K 帧的
    instance,cached_anchor 的位置从来不会更新 → 下帧关联时拿过期位置算 IoU →
    IoU 跌到阈值以下 → 关联失败 → 分新 ID(IDS+1)。
    
    Future Reasoning 给每个 cached query 一个"motion prediction MLP",让 cached
    anchor 在被遮挡期间按预测速度推进。重新可见时,cached_anchor 位置接近真实位置,
    IoU 关联成功 → 维持原 ID(IDS 不 +1)。

Implementation notes (跟原 PF-Track 的差异):
    - PF-Track 用一个 transformer 处理 history embeds (B, hist_len, D) 输出
      future_embeds (B, fut_len, D);我们简化成 MLP(input = concat(hist_embeds),
      output = (fut_len, 3) 的 (dx, dy, dz) offset)。
    - PF-Track 每个 instance 是 ``Instances`` 容器对象,我们用批量 tensor
      (B, num_temp, ...) 表示,所有计算 vectorize。
    - PF-Track 同时预测 future bbox / future logits;我们只预测位置 offset(因为
      Sparse4D 的关联只用位置 + class score)。
    - hist_len 默认 3,fut_len 默认 1(只预测下 1 帧,对 nuScenes 2Hz 已够)。
      Phase 2 可以增大到 fut_len=8 配合"motion 持续维持长遮挡 inactive track"。

Memory overhead:
    每个 cached query 多维护 hist_len × embed_dims float32 的历史 embed deque,
    num_temp=600 × hist_len=3 × 256 × 4B = 1.8 MB per scene,可忽略。
"""
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from .memory_bank import MemoryBank

__all__ = ["PFTrackInstanceBank"]


@PLUGIN_LAYERS.register_module()
class PFTrackInstanceBank(MemoryBank):
    """MemoryBank + PF-Track Future Reasoning (motion prediction).

    继承自 ``MemoryBank`` 而非 ``InstanceBank`` —— 这样 A_full config 可以同时启用
    A2 (memory pool) 和 A4 (future reasoning):
        - A4 alone (leave-out A2): num_memory_instances = num_temp_instances
          (退化成 ring buffer 行为) + future_reasoning_enable=True
        - A_full (A2 + A4): num_memory_instances = 512 + future_reasoning_enable=True
    leave-out A4 时直接用 ``MemoryBank``,leave-out A2 时用此类配 num_memory=num_temp。

    Args:
        hist_len: 历史帧数 (默认 3,跟 PF-Track 论文一致)
        fut_len: 预测未来帧数 (默认 1,nuScenes 2Hz 单步预测就足够)
        motion_mlp_hidden: motion prediction MLP 隐藏层维度 (默认 256)
        future_reasoning_enable: 是否启用 future reasoning (Bool flag,
            方便 leave-one-out ablation 时关掉 — 此时整个 bank 退化成
            普通 InstanceBank,memory 开销和数字都跟 baseline 一致)
        其他参数同 ``InstanceBank``。
    """

    def __init__(
        self,
        hist_len: int = 3,
        fut_len: int = 1,
        motion_mlp_hidden: int = 256,
        future_reasoning_enable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hist_len = int(hist_len)
        self.fut_len = int(fut_len)
        self.future_reasoning_enable = bool(future_reasoning_enable)

        if self.future_reasoning_enable:
            # Motion prediction MLP: 输入 hist_len 帧的 query embed (concat),
            # 输出 fut_len 帧的 (dx, dy, dz) offset
            in_dim = self.hist_len * self.embed_dims
            out_dim = self.fut_len * 3
            self.motion_mlp = nn.Sequential(
                nn.Linear(in_dim, motion_mlp_hidden),
                nn.LayerNorm(motion_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(motion_mlp_hidden, motion_mlp_hidden),
                nn.LayerNorm(motion_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(motion_mlp_hidden, out_dim),
            )
        else:
            self.motion_mlp = None

    def init_weight(self):
        super().init_weight()
        if self.motion_mlp is not None:
            for p in self.motion_mlp.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            # Last linear layer 输出 motion offset,初始化为接近 0
            # (避免训练初期 random offset 把 cached_anchor 推飞)
            last_linear = self.motion_mlp[-1]
            assert isinstance(last_linear, nn.Linear)
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def reset(self):
        super().reset()
        # history feature deque (B, hist_len, num_temp, D) 在 scene 切换时清空
        self.hist_feature_deque = None
        # 上一帧预测的 future offset (B, num_temp, fut_len, 3) — 下帧 get() 时用
        self.future_offset = None

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        """Cache + (optional) future motion prediction.

        Order:
            1. 父类正常 cache(更新 self.cached_feature / cached_anchor /
               confidence,topk num_temp_instances)
            2. 把新 cached_feature 推入 hist_feature_deque(deque 满后丢最老一帧)
            3. 如果 deque 已满(hist_len 帧都有),跑 motion_mlp 预测
               future_offset,缓存供下帧 get() 用
        """
        # Step 1: 标准 InstanceBank.cache 逻辑(更新 self.cached_feature / anchor)
        super().cache(instance_feature, anchor, confidence, metas, feature_maps)

        if not self.future_reasoning_enable or self.motion_mlp is None:
            return
        if self.cached_feature is None or self.cached_anchor is None:
            return

        # Step 2: 推入 history deque(shape: B, hist_len, num_temp, D)
        new_feat = self.cached_feature.detach().unsqueeze(1)  # (B, 1, num_temp, D)
        if (
            self.hist_feature_deque is None
            or self.hist_feature_deque.shape[0] != new_feat.shape[0]
            or self.hist_feature_deque.shape[2] != new_feat.shape[2]
        ):
            # 第一帧 / batch 或 num_temp 变了 → 重新填充 deque(每个槽都用当前帧填)
            # 这种填充策略下,前 hist_len 帧的 future_offset 不准,但比留 zero 好
            self.hist_feature_deque = new_feat.expand(
                -1, self.hist_len, -1, -1
            ).contiguous()
        else:
            # 正常 sliding: 丢最老 + 推新
            self.hist_feature_deque = torch.cat(
                [self.hist_feature_deque[:, 1:], new_feat], dim=1
            )

        # Step 3: 跑 motion MLP 预测 future offset
        # input: (B, num_temp, hist_len * D);output: (B, num_temp, fut_len * 3)
        B, hist_len, num_temp, D = self.hist_feature_deque.shape
        # Permute to (B, num_temp, hist_len, D),flatten last 2 dims to feed MLP
        hist_seq = self.hist_feature_deque.permute(0, 2, 1, 3).reshape(
            B, num_temp, hist_len * D
        )
        future_offset = self.motion_mlp(hist_seq)  # (B, num_temp, fut_len * 3)
        self.future_offset = future_offset.view(B, num_temp, self.fut_len, 3)

    def get(self, batch_size, metas=None, dn_metas=None):
        """Override get() — 在父类返回 cached_anchor 之前,apply future_offset."""
        # Step 1: 标准 InstanceBank.get(返回 cached_feature / cached_anchor)
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = super().get(batch_size, metas=metas, dn_metas=dn_metas)

        # Step 2: 在 cached_anchor 的 (x, y, z) 上加 future_offset(只用 fut_len 0
        # 这个槽,即下 1 帧的预测,因为 InstanceBank.get() 是为"下一帧的 get"提供
        # cache 的入口)
        if (
            self.future_reasoning_enable
            and temp_anchor is not None
            and self.future_offset is not None
            and self.future_offset.shape[0] == temp_anchor.shape[0]
            and self.future_offset.shape[1] == temp_anchor.shape[1]
        ):
            offset_xyz = self.future_offset[:, :, 0, :]  # (B, num_temp, 3)
            new_temp_anchor = temp_anchor.clone()
            new_temp_anchor[..., :3] = temp_anchor[..., :3] + offset_xyz
            temp_anchor = new_temp_anchor

        return (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        )
