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
        """Cache + history deque update.

        Order:
            1. 父类正常 cache(更新 self.cached_feature / cached_anchor /
               confidence,topk num_temp_instances)
            2. 把新 cached_feature 推入 hist_feature_deque
               (deque 满后丢最老一帧)

        注意: motion_mlp 的调用不在这里,而在下一帧的 get() 里 — 否则
        cache() 这一次的 autograd graph 在 loss_{t} backward 后被清空,
        等到 loss_{t+1} backward 时 motion_mlp 已经不在当前 graph,
        motion_mlp.weights 永远拿不到 gradient(DDP find_unused_parameters
        会报错)。
        """
        # Step 1: 标准 InstanceBank.cache 逻辑(更新 self.cached_feature / anchor)
        super().cache(instance_feature, anchor, confidence, metas, feature_maps)

        if not self.future_reasoning_enable or self.motion_mlp is None:
            return
        if self.cached_feature is None:
            return

        # Step 2: 推入 history deque(shape: B, hist_len, num_temp, D)
        # cached_feature 已经在父类 cache 里被 detach,这里直接用即可。
        new_feat = self.cached_feature.unsqueeze(1)  # (B, 1, num_temp, D)
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

    def _predict_future_offset(self, batch_size, device, dtype):
        """Run motion_mlp on hist_feature_deque, called from get() 而非 cache()
        以确保 motion_mlp 在当前 frame 的 autograd graph 里。

        DDP-critical:即使第一个 iter (deque 还没填充)也必须调用 motion_mlp,
        否则 DDP find_unused_parameters=False 会抱怨 motion_mlp.weights 没收到
        gradient。第一次调用时用 dummy zero 输入,output 也是 zero,加到
        temp_anchor 上不改变其值(可视为 no-op),但 motion_mlp 在 graph 里。

        Returns:
            future_offset: (B, num_temp, fut_len, 3) — 下一帧的 motion offset。
            首帧返回 zero tensor 但 motion_mlp 在 graph 里。
        """
        if self.motion_mlp is None:
            return None
        if self.hist_feature_deque is None:
            # 首帧:用 zero dummy 输入,确保 motion_mlp 在 autograd graph 里
            num_temp = self.num_temp_instances
            hist_seq = torch.zeros(
                batch_size, num_temp, self.hist_len * self.embed_dims,
                device=device, dtype=dtype,
            )
        else:
            B, hist_len, num_temp, D = self.hist_feature_deque.shape
            # Permute to (B, num_temp, hist_len, D),flatten last 2 dims to feed MLP
            hist_seq = self.hist_feature_deque.permute(0, 2, 1, 3).reshape(
                B, num_temp, hist_len * D
            )
        future_offset = self.motion_mlp(hist_seq)  # (B, num_temp, fut_len * 3)
        return future_offset.view(future_offset.shape[0], -1, self.fut_len, 3)

    def get(self, batch_size, metas=None, dn_metas=None):
        """Override get() — 调用 motion_mlp 算 future_offset,apply 到 temp_anchor.

        关键设计: motion_mlp 在这里调用(而非 cache() 里),这样 motion_mlp
        的 output future_offset 进入当前 frame 的 autograd graph;temp_anchor
        被作用 future_offset 后进入 head.forward 的 decoder,最终参与 loss。
        loss backward 时 gradient 沿 temp_anchor → future_offset → motion_mlp
        反传,motion_mlp.weights 能正确获得 gradient。
        """
        # Step 1: 标准 InstanceBank.get(返回 cached_feature / cached_anchor)
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = super().get(batch_size, metas=metas, dn_metas=dn_metas)

        # Step 2: 跑 motion_mlp 算 future_offset (在当前 frame 的 autograd graph 里)
        # DDP-critical:即使 temp_anchor 是 None(第一帧无 history),也调用
        # motion_mlp,否则 DDP find_unused_parameters=False 会抱怨。
        if self.future_reasoning_enable and self.motion_mlp is not None:
            # 选择正确 device/dtype:优先 anchor (current frame),fallback instance_feature
            ref = anchor if anchor is not None else instance_feature
            future_offset = self._predict_future_offset(
                batch_size=ref.shape[0],
                device=ref.device,
                dtype=ref.dtype,
            )
            if (
                future_offset is not None
                and temp_anchor is not None
                and future_offset.shape[1] == temp_anchor.shape[1]
            ):
                # 只用 fut_len 第 0 个槽(下 1 帧预测)
                offset_xyz = future_offset[:, :, 0, :]  # (B, num_temp, 3)
                new_temp_anchor = temp_anchor.clone()
                new_temp_anchor[..., :3] = temp_anchor[..., :3] + offset_xyz
                temp_anchor = new_temp_anchor
                # 缓存,便于 debug / eval-time 可视化(无 grad 需求)
                self.future_offset = future_offset.detach()
            elif future_offset is not None:
                # temp_anchor 是 None(第一帧无 history),但 motion_mlp 还是要参与
                # graph。把 future_offset.sum() * 0 加到 instance_feature 上做
                # zero-multiplied bridge,确保 motion_mlp.weights 通过 instance_feature
                # → decoder → loss 链路有 grad path(grad = 0 但路径存在)。
                instance_feature = instance_feature + future_offset.sum() * 0.0

        return (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        )
