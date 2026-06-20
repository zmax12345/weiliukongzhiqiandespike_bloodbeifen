import os
import glob
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
import MinkowskiEngine as ME


class FlexibleBloodFlowDataset(Dataset):
    def __init__(self, data_config, mask_path="hot_pixel_mask.npy", T=1, seq_len=5000, dt_us=20,
                 max_velocity=None, event_norm_mode="none", event_norm_stats=None,
                 event_norm_reference_mean=None, event_norm_eps=1e-6, event_norm_clip=(0.25, 4.0),
                 event_intensity_jitter_range=None, event_norm_seed=1234, return_metadata=False):
        """
        data_config: 字典格式。支持传文件夹路径，也支持传特定 csv 文件路径。
                     例如 {"/data/zm/.../": 0.0105, "/data/zm/.../1.0mm_clip.csv": 0.0105}
        T: 每个切片的 bin 数量。为了保留 20us 分辨率，T 设为 1
        seq_len: 一次前向传播的总时间步数 (例如 100ms 对应 5000 步)
        dt_us: 基础时间间隔 20 微秒
        max_velocity: 可选，只加载流速 <= max_velocity 的样本
        event_norm_mode: "none", "source_scale", or "sample_scale".
                         source_scale uses unlabeled source-level event-count statistics.
        """
        self.data_config = data_config
        self.T = T
        self.seq_len = seq_len
        self.dt = dt_us
        self.max_velocity = max_velocity
        self.event_norm_mode = event_norm_mode
        self.event_norm_stats = event_norm_stats or {}
        self.event_norm_reference_mean = event_norm_reference_mean
        self.event_norm_eps = event_norm_eps
        self.event_norm_clip = event_norm_clip
        self.event_intensity_jitter_range = event_intensity_jitter_range
        self.event_norm_seed = event_norm_seed
        self.return_metadata = return_metadata
        self.source_sample_counts = {}
        self.file_sample_counts = {}
        self.velocity_sample_counts = {}
        self.source_sample_indices = {}
        self.velocity_sample_indices = {}
        self.source_velocity_sample_indices = {}
        self.sample_metadata = []
        self.event_norm_summary = {}
        self.source_to_id = {}
        self.source_env_maps = {}

        self.hot_mask = np.load(mask_path) if os.path.exists(mask_path) else np.zeros((800, 1280), dtype=bool)
        self.samples = self._build_dataset()

    @staticmethod
    def _normalize_env_map(count_map):
        if not np.any(count_map > 0):
            return torch.zeros((1, 100, 368), dtype=torch.float32)
        positive = count_map[count_map > 0]
        scale = float(np.percentile(positive, 99.0))
        if scale <= 0 or not np.isfinite(scale):
            scale = float(positive.max())
        if scale <= 0 or not np.isfinite(scale):
            return torch.zeros((1, 100, 368), dtype=torch.float32)
        normalized = np.clip(count_map.astype(np.float32) / scale, 0.0, 1.0)
        return torch.from_numpy(normalized).unsqueeze(0).to(torch.float32)

    def _resolve_reference_mean(self, raw_total_events):
        if self.event_norm_reference_mean is not None:
            return float(self.event_norm_reference_mean)
        if self.event_norm_stats and self.event_norm_stats.get("reference_mean_events_per_sample") is not None:
            return float(self.event_norm_stats["reference_mean_events_per_sample"])
        if len(raw_total_events) == 0:
            return 0.0
        return float(np.mean(raw_total_events))

    def _clip_scale(self, scale):
        if self.event_norm_clip is None:
            return float(scale)
        clip_min, clip_max = self.event_norm_clip
        return float(np.clip(scale, clip_min, clip_max))

    def _build_event_norm_summary(self, sample_records, source_stats, source_scales, reference_mean):
        raw = np.array([r["raw_total_events"] for r in sample_records], dtype=np.float64)
        normalized = np.array(
            [r["raw_total_events"] * r["event_scale"] for r in sample_records],
            dtype=np.float64,
        )

        def _stats(values):
            if values.size == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        return {
            "event_norm_mode": self.event_norm_mode,
            "reference_mean_events_per_sample": float(reference_mean),
            "source_stats": source_stats,
            "source_scales": source_scales,
            "event_norm_clip": self.event_norm_clip,
            "event_intensity_jitter_range": self.event_intensity_jitter_range,
            "num_samples": len(sample_records),
            "raw_total_events": _stats(raw),
            "normalized_total_events": _stats(normalized),
        }

    def _build_dataset(self):
        if self.event_norm_mode not in {"none", "source_scale", "sample_scale"}:
            raise ValueError(f"Unsupported event_norm_mode: {self.event_norm_mode}")

        sample_records = []
        source_counts = defaultdict(int)
        file_counts = defaultdict(int)
        velocity_counts = defaultdict(int)
        source_indices = defaultdict(list)
        velocity_indices = defaultdict(list)
        source_velocity_indices = defaultdict(lambda: defaultdict(list))
        source_env_counts = defaultdict(lambda: np.zeros((100, 368), dtype=np.float64))
        for path, d_val in self.data_config.items():
            if path not in self.source_to_id:
                self.source_to_id[path] = len(self.source_to_id)
            # 兼容文件夹或单个特定的 CSV 文件
            if os.path.isdir(path):
                csv_files = glob.glob(os.path.join(path, "*_clip.csv"))
            elif os.path.isfile(path) and path.endswith('.csv'):
                csv_files = [path]
            else:
                continue

            for file in csv_files:
                filename = os.path.basename(file)
                try:
                    # 针对 "1.0mm_clip.csv" 提取真值 1.0
                    v_true = float(filename.split('mm')[0])
                except ValueError:
                    print(f"警告: 无法从文件名 {filename} 提取流速，已跳过。")
                    continue

                if self.max_velocity is not None and v_true > self.max_velocity:
                    continue

                # 严格按照原版 Sparse-PINN 的 pandas 读取与清洗逻辑
                try:
                    df = pd.read_csv(file, header=None, names=['row', 'col', 't_in', 't_off'],
                                     dtype={'row': np.int32, 'col': np.int32, 't_in': np.int64, 't_off': np.int64},
                                     on_bad_lines='skip')
                except Exception as e:
                    print(f"警告: 读取 {filename} 失败，可能文件损坏。错误: {e}")
                    continue

                if df.empty: continue

                # 1. 限制为您指定的新 ROI
                df = df[(df['row'] >= 400) & (df['row'] <= 499) & (df['col'] >= 700) & (df['col'] <= 1067)].copy()
                if df.empty: continue

                # 过滤坏点
                valid_events = ~self.hot_mask[df['row'].values, df['col'].values]
                df = df[valid_events].copy()
                if df.empty: continue

                # 2. 坐标平移，匹配网络输入尺寸 100x368
                df['row'] = df['row'] - 400
                df['col'] = df['col'] - 700  # 注意：这里必须改成减去 700，让列坐标从 0 开始

                # 时间对齐与量化
                t_start = df['t_in'].min()
                df['t_bin'] = (df['t_in'] - t_start) // self.dt

                max_bin = df['t_bin'].max()
                total_frames = int(max_bin // self.T) + 1

                # 提取时间切片序列
                # 设定滑动步长为 1000 步 (即 20ms 滑动一次)，这样截取的样本既有重叠又能提供新信息
                stride = 1000
                extracted_count = 0
                max_samples_per_file = 10  # 限制每个 CSV 最多提取 5 个样本，防止内存炸裂

                # 提取时间切片序列 (加入 stride 滑动步长)
                for seq_start_idx in range(0, total_frames - self.seq_len + 1, stride):
                    frame_coords = []
                    start_bin = seq_start_idx * self.T
                    end_bin = (seq_start_idx + self.seq_len) * self.T

                    seq_df = df[(df['t_bin'] >= start_bin) & (df['t_bin'] < end_bin)]
                    raw_total_events = 0

                    for f_idx in range(self.seq_len):
                        frame_start_bin = start_bin + f_idx * self.T
                        frame_df = seq_df[
                            (seq_df['t_bin'] >= frame_start_bin) & (seq_df['t_bin'] < frame_start_bin + self.T)]

                        if len(frame_df) == 0:
                            coords = torch.empty((0, 2), dtype=torch.int32)
                        else:
                            locations = torch.IntTensor(
                                np.column_stack((frame_df['row'].values, frame_df['col'].values)))
                            features = torch.ones((len(frame_df), 1), dtype=torch.float32)
                            coords, _ = ME.utils.sparse_quantize(coordinates=locations, features=features,
                                                                 quantization_size=[1, 1])
                        raw_total_events += int(coords.shape[0])
                        frame_coords.append(coords)

                    sample_idx = len(sample_records)
                    sample_records.append(
                        {
                            "frame_coords": frame_coords,
                            "source_path": path,
                            "file_path": file,
                            "v_true": v_true,
                            "d_val": d_val,
                            "raw_total_events": raw_total_events,
                            "seq_start_idx": seq_start_idx,
                        }
                    )
                    extracted_count += 1
                    source_counts[path] += 1
                    file_counts[file] += 1
                    velocity_counts[v_true] += 1
                    source_indices[path].append(sample_idx)
                    velocity_indices[v_true].append(sample_idx)
                    source_velocity_indices[path][v_true].append(sample_idx)

                    # 达到设定的样本上限后，再跳出当前文件的读取
                    if extracted_count >= max_samples_per_file:
                        break

        for record in sample_records:
            for coords in record["frame_coords"]:
                if coords.shape[0] > 0:
                    rows = coords[:, 0].numpy()
                    cols = coords[:, 1].numpy()
                    np.add.at(source_env_counts[record["source_path"]], (rows, cols), 1.0)

        raw_total_events = np.array([r["raw_total_events"] for r in sample_records], dtype=np.float64)
        reference_mean = self._resolve_reference_mean(raw_total_events)

        source_stats = {}
        source_scales = {}
        for source, indices in source_indices.items():
            values = np.array([sample_records[i]["raw_total_events"] for i in indices], dtype=np.float64)
            source_mean = float(values.mean()) if values.size else 0.0
            source_stats[source] = {
                "mean": source_mean,
                "std": float(values.std()) if values.size else 0.0,
                "min": float(values.min()) if values.size else 0.0,
                "max": float(values.max()) if values.size else 0.0,
                "num_samples": int(values.size),
            }
            source_scales[source] = self._clip_scale(reference_mean / (source_mean + self.event_norm_eps)) if source_mean > 0 else 1.0

        self.source_env_maps = {
            source: self._normalize_env_map(count_map)
            for source, count_map in source_env_counts.items()
        }

        rng = np.random.default_rng(self.event_norm_seed)
        samples = []
        self.sample_metadata = []
        for record in sample_records:
            source_scale = source_scales.get(record["source_path"], 1.0)
            sample_scale = self._clip_scale(reference_mean / (record["raw_total_events"] + self.event_norm_eps)) if record["raw_total_events"] > 0 else 1.0

            if self.event_norm_mode == "source_scale":
                event_scale = source_scale
            elif self.event_norm_mode == "sample_scale":
                event_scale = sample_scale
            else:
                event_scale = 1.0

            if self.event_intensity_jitter_range is not None:
                jitter_low, jitter_high = self.event_intensity_jitter_range
                event_scale *= float(rng.uniform(jitter_low, jitter_high))

            sequence_data = []
            for coords in record["frame_coords"]:
                if coords.shape[0] == 0:
                    feats = torch.empty((0, 1), dtype=torch.float32)
                else:
                    # Source-level multiplicative event intensity normalization.
                    # Coordinates and timing stay unchanged; only event feature amplitude is scaled.
                    feats = torch.full((coords.shape[0], 1), float(event_scale), dtype=torch.float32)
                sequence_data.append((coords, feats))

            source_id = self.source_to_id.get(record["source_path"], -1)
            env_map = self.source_env_maps.get(
                record["source_path"],
                torch.zeros((1, 100, 368), dtype=torch.float32),
            )

            samples.append((sequence_data, record["v_true"], record["d_val"], env_map, source_id))
            self.sample_metadata.append(
                {
                    "source_path": record["source_path"],
                    "file_path": record["file_path"],
                    "v_true": record["v_true"],
                    "d_val": record["d_val"],
                    "source_id": source_id,
                    "raw_total_events": record["raw_total_events"],
                    "seq_start_idx": record["seq_start_idx"],
                    "source_scale": source_scale,
                    "sample_scale": sample_scale,
                    "event_scale": event_scale,
                    "final_event_scale_mean": event_scale,
                    "normalized_total_events_est": record["raw_total_events"] * event_scale,
                }
            )

        self.source_sample_counts = dict(sorted(source_counts.items()))
        self.file_sample_counts = dict(sorted(file_counts.items()))
        self.velocity_sample_counts = dict(sorted(velocity_counts.items()))
        self.source_sample_indices = dict(sorted(source_indices.items()))
        self.velocity_sample_indices = dict(sorted(velocity_indices.items()))
        self.source_velocity_sample_indices = {
            source: dict(sorted(velocity_map.items()))
            for source, velocity_map in sorted(source_velocity_indices.items())
        }
        self.event_norm_summary = self._build_event_norm_summary(
            self.sample_metadata,
            source_stats,
            source_scales,
            reference_mean,
        )
        return samples

    def get_event_norm_summary(self):
        return self.event_norm_summary

    def get_reference_event_norm_stats(self):
        return {
            "reference_mean_events_per_sample": self.event_norm_summary.get("reference_mean_events_per_sample", 0.0),
            "source_stats": self.event_norm_summary.get("source_stats", {}),
            "source_scales": self.event_norm_summary.get("source_scales", {}),
            "event_norm_mode": self.event_norm_summary.get("event_norm_mode", self.event_norm_mode),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.return_metadata:
            return (*self.samples[index], self.sample_metadata[index])
        return self.samples[index]


def sequence_sparse_collate(batch):
    seq_len = len(batch[0][0])
    has_metadata = len(batch[0]) > 5
    batched_seq_data = []

    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in batch]
        feats_t = [sample[0][t][1] for sample in batch]
        b_coords, b_feats = ME.utils.sparse_collate(coords_t, feats_t)
        batched_seq_data.append((b_coords, b_feats))

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    d_values = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
    env_maps = torch.stack([sample[3] for sample in batch], dim=0)
    source_ids = torch.tensor([sample[4] for sample in batch], dtype=torch.long)

    if has_metadata:
        metadata = [sample[5] for sample in batch]
        return batched_seq_data, labels, d_values, env_maps, source_ids, metadata
    return batched_seq_data, labels, d_values, env_maps, source_ids
