#!/bin/bash
# =============================================================================
# Sparse4D-V3 数据 + 预训练权重 + pkl + anchor 一键准备
# =============================================================================
# 产出:
#   data/nuscenes              -> /mnt/datasets/nuscenes/v1.0.0          (symlink)
#   data/nuscenes_anno_pkls/   -> 训练/验证 pkl 标注
#   ckpt/resnet50-19c8e357.pth -> R50 backbone ImageNet 预训练
#   ckpt/sparse4dv3_r50.pth    -> 作者放出的 Sparse4Dv3 R50 完整 ckpt
#   nuscenes_kmeans900.npy     -> k-means 生成的 900 anchor(放工作目录根部)
# =============================================================================
set -euo pipefail

SPARSE4D_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
NUSCENES_DIR=/mnt/datasets/nuscenes/v1.0.0
cd "${SPARSE4D_ROOT}"

# shellcheck disable=SC1091
source "${SPARSE4D_ROOT}/.venv/bin/activate"

# --------------------------------------------------------------------------- #
# 1. 创建 data/nuscenes 软链接
# --------------------------------------------------------------------------- #
mkdir -p data
if [ ! -e data/nuscenes ]; then
    ln -s "${NUSCENES_DIR}" data/nuscenes
    echo "[OK] data/nuscenes -> ${NUSCENES_DIR}"
else
    echo "[SKIP] data/nuscenes already exists"
fi

# 验证数据集结构
for sub in samples sweeps v1.0-trainval maps; do
    if [ ! -d "data/nuscenes/${sub}" ]; then
        echo "[FAIL] data/nuscenes/${sub} not found!"
        exit 1
    fi
done
echo "[OK] nuScenes data root layout verified"

# --------------------------------------------------------------------------- #
# 2. 下载 ResNet50 ImageNet 预训练权重
# --------------------------------------------------------------------------- #
mkdir -p ckpt
if [ ! -f ckpt/resnet50-19c8e357.pth ]; then
    wget --no-check-certificate \
        https://download.pytorch.org/models/resnet50-19c8e357.pth \
        -O ckpt/resnet50-19c8e357.pth
fi
echo "[OK] ckpt/resnet50-19c8e357.pth ($(du -h ckpt/resnet50-19c8e357.pth | cut -f1))"

# --------------------------------------------------------------------------- #
# 3. 下载作者放出的 Sparse4Dv3 R50 完整 ckpt(用于评测 pipeline 对齐)
# --------------------------------------------------------------------------- #
if [ ! -f ckpt/sparse4dv3_r50.pth ]; then
    wget --no-check-certificate \
        https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth \
        -O ckpt/sparse4dv3_r50.pth
fi
echo "[OK] ckpt/sparse4dv3_r50.pth ($(du -h ckpt/sparse4dv3_r50.pth | cut -f1))"

# --------------------------------------------------------------------------- #
# 4. 生成 nuScenes pkl(包含 cam/lidar 元信息 + GT)
# --------------------------------------------------------------------------- #
PKL_DIR=data/nuscenes_anno_pkls
mkdir -p "${PKL_DIR}"

if [ ! -f "${PKL_DIR}/nuscenes_infos_train.pkl" ] \
    || [ ! -f "${PKL_DIR}/nuscenes_infos_val.pkl" ]; then
    echo "[INFO] Generating pkl files (~20 min, please wait)..."
    python tools/nuscenes_converter.py \
        --root_path data/nuscenes \
        --version v1.0-trainval \
        --info_prefix "${PKL_DIR}/nuscenes" \
        --max_sweeps 10
else
    echo "[SKIP] pkl files already exist"
fi
echo "[OK] pkl: train=$(du -h ${PKL_DIR}/nuscenes_infos_train.pkl | cut -f1), val=$(du -h ${PKL_DIR}/nuscenes_infos_val.pkl | cut -f1)"

# --------------------------------------------------------------------------- #
# 5. 生成 k-means anchor(必须放在工作目录根部,config 用相对路径加载)
# --------------------------------------------------------------------------- #
if [ ! -f nuscenes_kmeans900.npy ]; then
    echo "[INFO] Running k-means (n=900, may take ~10 min)..."
    # anchor_generator.py 用了 from projects.mmdet3d_plugin.core.box3d
    # 必须把项目根目录加入 PYTHONPATH
    PYTHONPATH=${SPARSE4D_ROOT}:${PYTHONPATH:-} python tools/anchor_generator.py \
        --ann_file "${PKL_DIR}/nuscenes_infos_train.pkl" \
        --num_anchor 900 \
        --detection_range 55 \
        --output_file_name nuscenes_kmeans900.npy
else
    echo "[SKIP] nuscenes_kmeans900.npy already exists"
fi
echo "[OK] anchor: $(du -h nuscenes_kmeans900.npy | cut -f1)"

echo ""
echo "[DONE] Data + checkpoint + pkl + anchor all ready."
echo "   Train ckpt : ckpt/resnet50-19c8e357.pth"
echo "   Author ckpt: ckpt/sparse4dv3_r50.pth"
echo "   Pkl dir    : ${PKL_DIR}/"
echo "   Anchor file: nuscenes_kmeans900.npy"
