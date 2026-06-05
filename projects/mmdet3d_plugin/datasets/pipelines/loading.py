import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        results["points"] = points
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweepsSparse4D(object):
    """Load and accumulate LiDAR points from multiple sweeps (Phase 2 F1).

    跟 mmdet3d 的 ``LoadPointsFromMultiSweeps`` 等价,但**直接处理 ndarray 格式**
    (兼容 sparse4d 自己的 LoadPointsFromFile 输出),不依赖 BasePoints 抽象。

    每个 sample 累积 ``sweeps_num`` 个 sweep 的点云,通过 ego motion 变换对齐到
    当前 sample 的 LiDAR 坐标系。输出 (N_total, 5) ndarray,5 维 =
    (x, y, z, intensity, sweep_time_offset)。

    Args:
        sweeps_num (int): 要累积的 sweep 数量(默认 10,nuScenes 标准 setting)
        load_dim (int): 每个 sweep pcd.bin 文件里的维度数(默认 5)
        use_dim (list): 要用的维度索引(默认 [0,1,2,3,4])
        time_dim (int): 时间偏移要写到哪一维(默认 4)
        pad_empty_sweeps (bool): 当 sample 没有 sweep 时(scene 起点),是否
            把当前帧 padding 到 sweeps_num 份(避免 voxel layer 在不同 sample
            收到不同点云规模导致的 batch size mismatch)
        remove_close (bool): 是否过滤距离 ego < 1m 的点(常见做法,这些点是
            ego 自己的车顶反射,不是真物体)
        test_mode (bool): 测试时是否取最早的 sweeps(deterministic);
            训练时随机选(增加多样性)
    """

    def __init__(
        self,
        sweeps_num: int = 10,
        load_dim: int = 5,
        use_dim: list = [0, 1, 2, 3, 4],
        time_dim: int = 4,
        pad_empty_sweeps: bool = False,
        remove_close: bool = False,
        test_mode: bool = False,
        file_client_args: dict = dict(backend="disk"),
    ):
        self.sweeps_num = int(sweeps_num)
        self.load_dim = int(load_dim)
        self.use_dim = list(use_dim)
        self.time_dim = int(time_dim)
        assert time_dim < load_dim, (
            f"time_dim ({time_dim}) must be < load_dim ({load_dim})"
        )
        self.pad_empty_sweeps = bool(pad_empty_sweeps)
        self.remove_close = bool(remove_close)
        self.test_mode = bool(test_mode)
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    @staticmethod
    def _remove_close(points, radius=1.0):
        """过滤距离 ego < radius 的点(车顶 self-reflection)。"""
        not_close = np.linalg.norm(points[:, :2], axis=1) >= radius
        return points[not_close]

    def __call__(self, results):
        points = results["points"]  # (N, use_dim) 当前 sample 点云,从 LoadPointsFromFile 来
        # 把 time_dim 维度清零(当前 sample 时间偏移 = 0)
        if points.shape[1] > self.time_dim:
            points[:, self.time_dim] = 0.0

        sweep_points_list = [points]
        # 单位约定(踩过坑,务必注意):NuScenes3DDetTrackDataset.get_data_info
        # 已经把 sample 的 timestamp 除以 1e6 转成"秒"(见 dataset 的
        # `timestamp=info["timestamp"] / 1e6`),所以这里**不能再除 1e6**。
        # sweep["timestamp"](下面 ts_sweep)才是 pkl 原始的微秒,需要各自 /1e6。
        # 旧 bug:这里多除了一次 → ts_cur≈1531 而 ts_sweep≈1.5e9,相减无法抵消
        # 绝对时间戳,导致 time_dim 被写入 ~1.5e9 的天文数字,撑爆 LiDAR
        # SparseEncoder 第一个 BN(running_var→1e17),训练 NaN / lidar-only 崩溃。
        ts_cur = results["timestamp"]  # 已是秒,不要再 /1e6

        sweeps = results.get("sweeps", [])

        # 选要累积哪些 sweep
        if self.pad_empty_sweeps and len(sweeps) == 0:
            # scene 起点没 prev sweep,用当前帧 padding(避免 batch size 不一致)
            for _ in range(self.sweeps_num):
                pad = self._remove_close(points) if self.remove_close else points.copy()
                sweep_points_list.append(pad)
        else:
            if len(sweeps) <= self.sweeps_num:
                choices = np.arange(len(sweeps))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(sweeps), self.sweeps_num, replace=False
                )

            for idx in choices:
                sweep = sweeps[idx]
                pts_path = sweep["data_path"]
                points_sweep = self._load_points(pts_path)
                points_sweep = points_sweep.reshape(-1, self.load_dim)[:, self.use_dim]

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)

                # ego motion 对齐:把 sweep 的点投到当前 sample 的 LiDAR 坐标系
                # sweep['sensor2lidar_rotation'] 和 'sensor2lidar_translation'
                # 是从 sweep LiDAR 到 current LiDAR 的变换
                R = sweep["sensor2lidar_rotation"]   # (3, 3)
                t = sweep["sensor2lidar_translation"]  # (3,)
                xyz = points_sweep[:, :3]
                xyz_aligned = xyz @ R.T + t
                points_sweep[:, :3] = xyz_aligned

                # 在 time_dim 写入相对当前帧的时间偏移(秒,负数表示过去)
                ts_sweep = sweep["timestamp"] / 1e6
                if points_sweep.shape[1] > self.time_dim:
                    points_sweep[:, self.time_dim] = ts_sweep - ts_cur

                sweep_points_list.append(points_sweep.astype(np.float32))

        # 累积所有 sweep 点云成一个大 ndarray
        all_points = np.concatenate(sweep_points_list, axis=0)
        results["points"] = all_points
        return results
