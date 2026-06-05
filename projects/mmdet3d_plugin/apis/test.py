# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util


def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order="F", dtype="uint8"
                )
            )[0]
        )  # encoded with RLE
    return [encoded_mask_results]


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if "bbox_results" in result.keys():
                    bbox_result = result["bbox_results"]
                    batch_size = len(result["bbox_results"])
                    bbox_results.extend(bbox_result)
                if (
                    "mask_results" in result.keys()
                    and result["mask_results"] is not None
                ):
                    mask_result = custom_encode_mask_results(
                        result["mask_results"]
                    )
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir + "_mask" if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(
                mask_results, len(dataset), tmpdir
            )
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
    return {"bbox_results": bbox_results, "mask_results": mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full(
            (MAX_LEN,), 32, dtype=torch.uint8, device="cuda"
        )
        if rank == 0:
            mmcv.mkdir_or_exist(".dist_test")
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f"part_{i}.pkl")
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        """
        bacause we change the sample of the evaluation stage to make sure that
        each gpu will handle continuous sample,
        """
        # for res in zip(*part_list):
        for res in part_list:
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """用 NCCL allgather 跨节点收集 eval 结果(不依赖共享文件系统)。

    背景:旧版本是个 stub `collect_results_cpu(result_part, size)`,实际走
    cpu collect 路径,要求 work_dir/.eval_hook 是跨节点共享盘。多机训练
    work_dir 在节点本地盘时,rank 0 看不到 rank>=N_per_node 写的 part 文件
    → FileNotFoundError(part_8.pkl)。本实现走 GPU allgather,无此问题。

    Args:
        result_part: 当前 rank 的 outputs(list of dict / 任意 picklable 对象)
        size: 整个 dataset 的样本数(用于裁掉 dataloader 的 padding)
    Returns:
        rank 0: 合并后的 ordered_results;其他 rank: None
    """
    rank, world_size = get_dist_info()
    # 1) 序列化为 byte tensor(放 GPU,因为 NCCL 只能跑 GPU tensor)
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )
    # 2) 先 allgather 各 rank 的字节长度(allgather 要求 shape 一致,所以先对齐)
    shape_tensor = torch.tensor([part_tensor.shape[0]], device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # 3) pad 当前 rank 到最大长度,然后 allgather 真实数据
    shape_max = torch.cat(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor.item()] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    dist.all_gather(part_recv_list, part_send)
    # 4) rank 0 反序列化 + 按 rank 顺序合并(每 rank 的 outputs 在内部已是该
    # rank 的连续样本顺序,见 collect_results_cpu 注释里说的 sampler 行为)
    if rank != 0:
        return None
    part_list = []
    for recv, shape in zip(part_recv_list, shape_list):
        part_bytes = recv[: shape.item()].cpu().numpy().tobytes()
        part_list.append(pickle.loads(part_bytes))
    ordered_results = []
    for res in part_list:
        ordered_results.extend(list(res))
    return ordered_results[:size]
