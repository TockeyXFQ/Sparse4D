import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    """Default InstanceBank + optional Birth/Death state machine (A5).

    Birth/Death state machine (Poly-MOT IROS 2023 风格,开源):
        当 ``birth_death`` 不为 None 时启用三态机替代默认 prev_id++ 暴力 ID 分配:
            * Tentative(试用期): 新出现 query 先进试用期,需要连续
              ``birth_min_hits`` 帧 score >= ``birth_score_thr`` 才转 Confirmed
            * Confirmed(确认): 正常 track,丢失 N 帧仍保留(由 InstanceBank 的
              num_temp_instances 容量自然决定生命周期)
            * Dead(死亡): 试用期失败 / 超期未观测 → 释放 ID,后续不再分配

        此 trick 不需要重训,纯 post-process 推理 trick (Phase 1 A5)。
        训练时(self.training==True)走默认 prev_id++ 路径,保持训练阶段
        instance_id 的稳定性供 dn / tracking loss 用。

    Args:
        birth_death: ``None`` (默认,等价 baseline) 或 dict 启用状态机:
            ``dict(min_hits=3, score_thr=0.4, max_age=5)``
            - min_hits: Tentative → Confirmed 需要的连续高置信帧数 (默认 3)
            - score_thr: 进入 Tentative 的 score 阈值 (默认 0.4)
            - max_age: Confirmed 丢失多少帧后转 Dead (默认 5)
    """

    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        birth_death=None,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        # ---- A5 Birth/Death state machine 配置 ----
        if birth_death is None:
            self.birth_death_cfg = None
        else:
            self.birth_death_cfg = dict(
                min_hits=int(birth_death.get("min_hits", 3)),
                score_thr=float(birth_death.get("score_thr", 0.4)),
                max_age=int(birth_death.get("max_age", 5)),
            )
        # state machine 运行时态: 以 instance_id 为 key,值为
        # dict(state='tentative'|'confirmed', hits=int, age_since_seen=int)
        # state machine 只在 eval 阶段启用,train 时清空
        self._track_states = {}

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0
        # A5: 切到新 scene 时清空 state machine 状态
        self._track_states = {}

    def get(self, batch_size, metas=None, dn_metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id, confidence)

        # ============ A5: Birth/Death 状态机 ============
        # 训练时跳过(state 由 sampler 控制),只在 eval 阶段启用。
        if (
            self.birth_death_cfg is not None
            and not self.training
            and threshold is not None
        ):
            instance_id = self._apply_birth_death(
                instance_id, confidence, threshold
            )
        return instance_id

    def _apply_birth_death(self, instance_id, confidence, threshold):
        """A5 - Tentative/Confirmed/Dead state machine.

        - Tentative (新出现): 连续 ``min_hits`` 帧 score >= ``score_thr`` 转 Confirmed
        - Confirmed (正常): 丢失帧累计 > ``max_age`` 转 Dead
        - Dead: ID 在本帧被 mask 掉 (instance_id=-1) → 等价于"丢这个 ID"

        Args:
            instance_id: (B, A) 当前帧分配好的 instance_id (含 -1 表示 BG)
            confidence: (B, A) max-class sigmoid score
            threshold: detection score threshold (Sparse4D 默认从
                       ``decoder.score_threshold`` 传入)

        Returns:
            instance_id: 同形状,但被 state machine 过滤后:
                - Tentative 未达 min_hits 的: 维持 ID 但下次 reappear 时仍 tentative
                - Dead 的: 被设为 -1 (相当于丢轨,Sparse4D 后续会自然不输出这个 track)
        """
        cfg = self.birth_death_cfg
        min_hits = cfg["min_hits"]
        score_thr = cfg["score_thr"]
        max_age = cfg["max_age"]

        instance_id_cpu = instance_id.detach().cpu().numpy()
        confidence_cpu = confidence.detach().cpu().numpy()

        bs, num_anchor = instance_id_cpu.shape
        new_instance_id = instance_id_cpu.copy()
        seen_ids_this_frame = set()

        for b in range(bs):
            for a in range(num_anchor):
                tid = int(instance_id_cpu[b, a])
                if tid < 0:
                    continue
                score = float(confidence_cpu[b, a])
                seen_ids_this_frame.add(tid)
                state = self._track_states.get(tid)
                if state is None:
                    self._track_states[tid] = dict(
                        state="tentative",
                        hits=1 if score >= score_thr else 0,
                        age_since_seen=0,
                    )
                    if min_hits > 1 and score < score_thr:
                        new_instance_id[b, a] = -1
                else:
                    state["age_since_seen"] = 0
                    if state["state"] == "tentative":
                        if score >= score_thr:
                            state["hits"] += 1
                            if state["hits"] >= min_hits:
                                state["state"] = "confirmed"
                        if state["state"] != "confirmed":
                            new_instance_id[b, a] = -1

        for tid in list(self._track_states.keys()):
            if tid in seen_ids_this_frame:
                continue
            state = self._track_states[tid]
            state["age_since_seen"] += 1
            if state["age_since_seen"] > max_age:
                state["state"] = "dead"
                del self._track_states[tid]

        return instance_id.new_tensor(new_instance_id)

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
