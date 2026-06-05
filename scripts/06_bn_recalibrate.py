#!/usr/bin/env python3
# =============================================================================
# LiDAR BatchNorm 重标定 (BN recalibration) —— 路线 1 诊断/修复
#
# 背景:expF1/expF_full 的 LiDAR SparseEncoder 第一个 BN(conv_input.1)的
# running_var 在训练早期不稳定期被污染(F1=NaN, F_full~1e17)。卷积权重本身
# 是好的,只是 BN 的 running_mean/var 这两个**非可学习 buffer**坏了。eval 时
# BN 用坏掉的 running stats 归一化 → LiDAR BEV 退化成常数图 → lidar-only 崩溃。
#
# 本脚本:整体 eval(),只把 LiDAR backbone(pts_*)的 BN 设成 train-mode 并
# 重置统计量,然后**纯前向**(no_grad)跑若干 batch,让这些 BN 用 cumulative
# moving average 重新累积干净的 running_mean/var,最后存成新 ckpt。
# 图像 backbone 的 BN 保持 eval、完全不动。可学习权重(conv/linear)全程不变。
# =============================================================================
import argparse
import copy
import importlib
import os

import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, save_checkpoint, wrap_fp16_model
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

# LiDAR backbone 的 4 个子模块前缀(只重标定这些里面的 BN)
LIDAR_PREFIXES = (
    "pts_voxel_encoder",
    "pts_middle_encoder",
    "pts_backbone",
    "pts_neck",
)
# 诊断用:重点观察这个被污染的 BN
WATCH_BN = "pts_middle_encoder.conv_input.1"


def parse_args():
    p = argparse.ArgumentParser(description="LiDAR BatchNorm recalibration")
    p.add_argument("config", help="config 文件路径")
    p.add_argument("checkpoint", help="待重标定的 ckpt 路径")
    p.add_argument(
        "--out",
        default=None,
        help="输出 ckpt 路径(默认在原 ckpt 同目录加 _bnrecalib 后缀)",
    )
    p.add_argument(
        "--num-batches",
        type=int,
        default=500,
        help="重标定用的 batch 数(默认 500;BN 统计量收敛快,几百个足够)",
    )
    p.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="重标定数据来源 split(默认 train,论文可用;val 仅快速诊断)",
    )
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="覆盖 config 字段(同一个 --cfg-options 后跟多个 key=value)",
    )
    return p.parse_args()


def load_plugin(cfg):
    """照搬 tools/test.py 的 plugin 加载逻辑,注册自定义模块。"""
    if hasattr(cfg, "plugin") and cfg.plugin and hasattr(cfg, "plugin_dir"):
        _module_dir = os.path.dirname(cfg.plugin_dir).split("/")
        _module_path = _module_dir[0]
        for m in _module_dir[1:]:
            _module_path = _module_path + "." + m
        importlib.import_module(_module_path)


def get_watch_stat(model):
    """返回被观察 BN 的 (running_var.max, |running_mean|.max),用于前后对比。"""
    for name, m in model.named_modules():
        if name == WATCH_BN and isinstance(m, _BatchNorm):
            return (
                m.running_var.max().item(),
                m.running_mean.abs().max().item(),
            )
    return None, None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    load_plugin(cfg)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    # ---- dataset: 用 test pipeline(无随机增强,匹配推理分布) ----
    # 默认把 ann_file 换成 train split(避免用 val 数据碰模型,论文更干净);
    # --split val 则直接用原 test 配置(零兼容风险,适合快速诊断)。
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.test_mode = True
    if args.split == "train":
        train_ann = cfg.data.train.get("ann_file", None)
        assert train_ann is not None, (
            "cfg.data.train 没有 ann_file,请改用 --split val"
        )
        test_cfg.ann_file = train_ann
        print(f">>> 重标定数据: train split ({train_ann})")
    else:
        print(f">>> 重标定数据: val split ({test_cfg.get('ann_file')})")

    dataset = build_dataset(test_cfg)
    # shuffle=False:test_mode dataset 没有 `flag` 属性,mmdet 的 GroupSampler
    # (shuffle=True 时启用)会 assert 失败。顺序取样本对 BN 统计量估计无影响
    # (数据本身跨多场景顺序排列,代表性足够)。
    data_loader = build_dataloader_origin(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 4),
        dist=False,
        shuffle=False,
    )

    # ---- model ----
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    real = model.module

    # ---- 选出 LiDAR backbone 的 BN 层 ----
    targets = [
        (name, m)
        for name, m in real.named_modules()
        if isinstance(m, _BatchNorm) and name.startswith(LIDAR_PREFIXES)
    ]
    assert targets, "没找到 LiDAR backbone 的 BN,检查 config 是否启用 LiDAR 分支"

    v0, m0 = get_watch_stat(real)
    print(f">>> 找到 {len(targets)} 个 LiDAR BN 层待重标定")
    print(f">>> 重标定前 {WATCH_BN}: running_var.max={v0:.3e} |mean|.max={m0:.3e}")

    # ---- 整体 eval,只让目标 BN 进 train-mode + 重置统计量 ----
    # momentum=None → cumulative moving average(N 个 batch 后是无偏平均)
    model.eval()
    for _, m in targets:
        m.reset_running_stats()
        m.momentum = None
        m.train()

    # ---- 纯前向累积统计量 ----
    print(f">>> 开始重标定,跑 {args.num_batches} 个 batch (no_grad)...")
    n = 0
    with torch.no_grad():
        for data in data_loader:
            # 复用 simple_test 前向(self.training=False),只有目标 BN 在
            # train-mode 会更新 running stats;masked-modal 不触发(仅 train 用)
            model(return_loss=False, rescale=True, **data)
            n += 1
            if n % 50 == 0:
                v, _ = get_watch_stat(real)
                print(f"    [{n}/{args.num_batches}] {WATCH_BN} var.max={v:.3e}")
            if n >= args.num_batches:
                break

    v1, m1 = get_watch_stat(real)
    print(f">>> 重标定后 {WATCH_BN}: running_var.max={v1:.3e} |mean|.max={m1:.3e}")
    print(f">>> 改善: var.max {v0:.3e} → {v1:.3e}")

    # ---- 存新 ckpt ----
    out = args.out
    if out is None:
        root, ext = os.path.splitext(args.checkpoint)
        out = f"{root}_bnrecalib{ext}"
    # save_checkpoint 存 BN 已更新的 state_dict;切回 eval 保证存的是干净状态
    model.eval()
    save_checkpoint(real, out)
    print(f"\n[DONE] 重标定后的 ckpt 已保存: {out}")
    print("       下一步用它重测 G2 lidar-only,看 AMOTA 是否复活。")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
