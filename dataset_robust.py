import os
import glob
import re
import warnings
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None


def sparse_quantize_coordinates(locations):
    if locations.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int32)
    if ME is not None:
        coords, _ = ME.utils.sparse_quantize(
            coordinates=locations,
            features=torch.ones((locations.shape[0], 1), dtype=torch.float32),
            quantization_size=[1, 1],
        )
        return coords.to(torch.int32)
    return torch.unique(locations.to(torch.int32), dim=0)


def sparse_collate_fallback(coords_list, feats_list):
    batched_coords = []
    batched_feats = []
    for batch_idx, (coords, feats) in enumerate(zip(coords_list, feats_list)):
        if coords.numel() == 0:
            continue
        batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
        batched_coords.append(torch.cat((batch_col, coords.to(torch.int32)), dim=1))
        batched_feats.append(feats.to(torch.float32))
    if not batched_coords:
        return torch.empty((0, 3), dtype=torch.int32), torch.empty((0, 1), dtype=torch.float32)
    return torch.cat(batched_coords, dim=0), torch.cat(batched_feats, dim=0)


class FlexibleBloodFlowDataset(Dataset):
    BETA_EPS = 1e-8
    ROI_ROW_START = 100
    ROI_ROW_END = 200
    ROI_COL_START = 0
    ROI_COL_END = 1200
    SPATIAL_SHAPE = (ROI_ROW_END - ROI_ROW_START, ROI_COL_END - ROI_COL_START)

    def __init__(self, data_config, mask_path="hot_pixel_mask.npy", T=1, seq_len=5000, dt_us=20,
                 max_velocity=None, event_norm_mode="none", event_norm_stats=None,
                 event_norm_reference_mean=None, event_norm_eps=1e-6, event_norm_clip=(0.25, 4.0),
                 event_intensity_jitter_range=None, event_norm_seed=1234, return_metadata=False,
                 include_velocities=None, exclude_velocities=None):
        """
        data_config: 瀛楀吀鏍煎紡銆傛敮鎸佷紶鏂囦欢澶硅矾寰勶紝涔熸敮鎸佷紶鐗瑰畾 csv 鏂囦欢璺緞銆?
                     渚嬪 {"/data/zm/.../": 0.0105, "/data/zm/.../1.0mm_clip.csv": 0.0105}
        T: 姣忎釜鍒囩墖鐨?bin 鏁伴噺銆備负浜嗕繚鐣?20us 鍒嗚鲸鐜囷紝T 璁句负 1
        seq_len: 涓€娆″墠鍚戜紶鎾殑鎬绘椂闂存鏁?(渚嬪 100ms 瀵瑰簲 5000 姝?
        dt_us: 鍩虹鏃堕棿闂撮殧 20 寰
        max_velocity: 鍙€夛紝鍙姞杞芥祦閫?<= max_velocity 鐨勬牱鏈?        event_norm_mode: "none", "source_scale", or "sample_scale".
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
        self.include_velocities = self._normalize_velocity_filter(include_velocities)
        self.exclude_velocities = self._normalize_velocity_filter(exclude_velocities)
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
        self.channel_mask_cache = {}
        self.source_channel_masks = {}
        self.channel_mask_summary = {}

        self.hot_mask = self._load_hot_mask(mask_path)
        self.samples = self._build_dataset()

    @staticmethod
    def _load_hot_mask(mask_path):
        if mask_path is None or str(mask_path).strip().lower() in {"", "none", "null", "false", "off"}:
            print("WARNING: hot pixel mask disabled; training will use all events.")
            return np.zeros((800, 1280), dtype=bool)
        if not os.path.exists(mask_path):
            print(f"WARNING: hot pixel mask not found: {mask_path}; training will use all events.")
            return np.zeros((800, 1280), dtype=bool)
        mask = np.load(mask_path)
        if mask.shape != (800, 1280):
            raise ValueError(f"hot pixel mask must have shape (800, 1280), got {mask.shape} from {mask_path}.")
        return mask.astype(bool)

    @staticmethod
    def _normalize_velocity_filter(values):
        if values is None:
            return None
        return {round(float(value), 6) for value in values}

    @staticmethod
    def _parse_boolish(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
            return False
        return default

    @staticmethod
    def _parse_velocity_from_filename(filename):
        lower = filename.lower()
        if "mm" in lower:
            return float(lower.split("mm")[0])
        stem = os.path.splitext(filename)[0]
        numbers = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", stem)
        if not numbers:
            raise ValueError(f"Cannot parse velocity from filename {filename!r}.")
        return float(numbers[-1])

    @classmethod
    def _parse_source_config(cls, path, config_value):
        if isinstance(config_value, dict):
            if "d_value" not in config_value:
                raise ValueError(f"data_config[{path!r}] is a dict but has no required 'd_value'.")
            d_val = float(config_value["d_value"])
            k_max = config_value.get("K_max", config_value.get("k_max"))
            if k_max is None:
                warnings.warn(
                    f"data_config[{path!r}] has no K_max; defaulting K_max=1.0 for backward compatibility.",
                    RuntimeWarning,
                )
                k_max = 1.0
            k_max = float(k_max)
            condition = str(config_value.get("condition", "unknown"))
            sub_condition = str(config_value.get("sub_condition", condition))
            phantom_flag = float(config_value.get("phantom_flag", -1.0))
            split_group = str(config_value.get("split_group", "train_val"))
            quality = str(config_value.get("quality", "unknown"))
            use_for_training = bool(config_value.get("use_for_training", split_group == "train_val"))
            channel_mask_path = str(config_value.get("channel_mask_path", "") or "")
            channel_mask_enabled = cls._parse_boolish(
                config_value.get("channel_mask_enabled", bool(channel_mask_path)),
                default=bool(channel_mask_path),
            )
            if "channel_mask_path" not in config_value and channel_mask_enabled:
                warnings.warn(
                    f"data_config[{path!r}] has channel_mask_enabled=True but no channel_mask_path; "
                    "channel mask disabled for this source.",
                    RuntimeWarning,
                )
                channel_mask_enabled = False
            elif "channel_mask_path" not in config_value:
                warnings.warn(
                    f"data_config[{path!r}] has no channel_mask_path; channel mask disabled for this source.",
                    RuntimeWarning,
                )
        else:
            d_val = float(config_value)
            k_max = 1.0
            condition = "unknown"
            sub_condition = "unknown"
            phantom_flag = -1.0
            split_group = "train_val"
            quality = "legacy"
            use_for_training = True
            channel_mask_path = ""
            channel_mask_enabled = False

        beta_max = float(k_max ** 2)
        log_beta_max = float(np.log(beta_max + cls.BETA_EPS))
        return {
            "d_val": d_val,
            "K_max": k_max,
            "beta_max": beta_max,
            "log_beta_max": log_beta_max,
            "condition": condition,
            "sub_condition": sub_condition,
            "phantom_flag": phantom_flag,
            "split_group": split_group,
            "quality": quality,
            "use_for_training": use_for_training,
            "channel_mask_enabled": channel_mask_enabled,
            "channel_mask_path": channel_mask_path,
        }

    @staticmethod
    def _normalize_env_map(count_map):
        if not np.any(count_map > 0):
            return torch.zeros((1, *FlexibleBloodFlowDataset.SPATIAL_SHAPE), dtype=torch.float32)
        positive = count_map[count_map > 0]
        scale = float(np.percentile(positive, 99.0))
        if scale <= 0 or not np.isfinite(scale):
            scale = float(positive.max())
        if scale <= 0 or not np.isfinite(scale):
            return torch.zeros((1, *FlexibleBloodFlowDataset.SPATIAL_SHAPE), dtype=torch.float32)
        normalized = np.clip(count_map.astype(np.float32) / scale, 0.0, 1.0)
        return torch.from_numpy(normalized).unsqueeze(0).to(torch.float32)

    def _init_channel_mask_for_source(self, source_path, source_config):
        mask_enabled = bool(source_config.get("channel_mask_enabled", False))
        mask_path = str(source_config.get("channel_mask_path", "") or "")
        disabled_tokens = {"", "none", "null", "false", "off"}
        if not mask_enabled or mask_path.strip().lower() in disabled_tokens:
            if mask_enabled and mask_path.strip().lower() in disabled_tokens:
                print(
                    f"WARNING: source `{source_path}` has channel_mask_enabled=True "
                    "but no channel_mask_path; channel mask disabled for this source."
                )
            self.source_channel_masks[source_path] = None
            self.channel_mask_summary[source_path] = {
                "source": source_path,
                "sub_condition": source_config.get("sub_condition", "unknown"),
                "channel_mask_enabled": False,
                "channel_mask_path": "",
                "channel_mask_area_pixels": float("nan"),
                "channel_mask_area_ratio": float("nan"),
                "events_before_channel_mask": 0,
                "events_after_channel_mask": 0,
                "channel_mask_retained_ratio": 1.0,
            }
            return None

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Channel mask file not found for source `{source_path}`: {mask_path}")

        if mask_path not in self.channel_mask_cache:
            mask = np.load(mask_path)
            if mask.shape != self.SPATIAL_SHAPE:
                raise ValueError(
                    f"channel mask shape mismatch for source `{source_path}`. "
                    f"mask_path={mask_path}, expected shape={self.SPATIAL_SHAPE}, actual shape={mask.shape}."
                )
            self.channel_mask_cache[mask_path] = mask.astype(bool)

        channel_mask = self.channel_mask_cache[mask_path]
        self.source_channel_masks[source_path] = channel_mask
        mask_area = int(channel_mask.sum())
        self.channel_mask_summary[source_path] = {
            "source": source_path,
            "sub_condition": source_config.get("sub_condition", "unknown"),
            "channel_mask_enabled": True,
            "channel_mask_path": mask_path,
            "channel_mask_area_pixels": mask_area,
            "channel_mask_area_ratio": float(mask_area / channel_mask.size),
            "events_before_channel_mask": 0,
            "events_after_channel_mask": 0,
            "channel_mask_retained_ratio": 0.0,
        }
        return channel_mask

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
        source_env_counts = defaultdict(lambda: np.zeros(self.SPATIAL_SHAPE, dtype=np.float64))
        source_channel_event_counts = defaultdict(lambda: {"before": 0, "after": 0})
        for path, config_value in self.data_config.items():
            source_config = self._parse_source_config(path, config_value)
            channel_mask = self._init_channel_mask_for_source(path, source_config)
            if path not in self.source_to_id:
                self.source_to_id[path] = len(self.source_to_id)
            # 鍏煎鏂囦欢澶规垨鍗曚釜鐗瑰畾鐨?CSV 鏂囦欢
            if glob.has_magic(path):
                csv_files = sorted(glob.glob(path))
            elif os.path.isdir(path):
                csv_files = sorted(
                    set(glob.glob(os.path.join(path, "*_clip.csv")))
                    | set(glob.glob(os.path.join(path, "*.csv")))
                )
            elif os.path.isfile(path) and path.endswith('.csv'):
                csv_files = [path]
            else:
                continue

            for file in csv_files:
                filename = os.path.basename(file)
                try:
                    # 閽堝 "1.0mm_clip.csv" 鎻愬彇鐪熷€?1.0
                    v_true = self._parse_velocity_from_filename(filename)
                except ValueError:
                    print(f"WARNING: could not parse velocity from filename {filename}; skipped.")
                    continue

                if self.max_velocity is not None and v_true > self.max_velocity:
                    continue
                rounded_velocity = round(float(v_true), 6)
                if self.include_velocities is not None and rounded_velocity not in self.include_velocities:
                    continue
                if self.exclude_velocities is not None and rounded_velocity in self.exclude_velocities:
                    continue

                # 涓ユ牸鎸夌収鍘熺増 Sparse-PINN 鐨?pandas 璇诲彇涓庢竻娲楅€昏緫
                try:
                    df = pd.read_csv(file, header=None, names=['row', 'col', 't_in', 't_off'],
                                     dtype={'row': np.int32, 'col': np.int32, 't_in': np.int64, 't_off': np.int64},
                                     on_bad_lines='skip')
                except Exception as e:
                    print(f"璀﹀憡: 璇诲彇 {filename} 澶辫触锛屽彲鑳芥枃浠舵崯鍧忋€傞敊璇? {e}")
                    continue

                if df.empty: continue

                # Event ROI for current 6.17 data: rows 100:200, cols 0:1200.
                df = df[
                    (df['row'] >= self.ROI_ROW_START)
                    & (df['row'] < self.ROI_ROW_END)
                    & (df['col'] >= self.ROI_COL_START)
                    & (df['col'] < self.ROI_COL_END)
                ].copy()
                if df.empty: continue

                # 杩囨护鍧忕偣
                valid_events = ~self.hot_mask[df['row'].values, df['col'].values]
                df = df[valid_events].copy()
                if df.empty: continue

                events_before_channel = int(len(df))
                source_channel_event_counts[path]["before"] += events_before_channel
                if channel_mask is not None:
                    row_local = df['row'].values - self.ROI_ROW_START
                    col_local = df['col'].values - self.ROI_COL_START
                    keep_channel = channel_mask[row_local, col_local]
                    df = df[keep_channel].copy()
                events_after_channel = int(len(df))
                source_channel_event_counts[path]["after"] += events_after_channel
                if df.empty: continue

                # Shift to local network coordinates.
                df['row'] = df['row'] - self.ROI_ROW_START
                df['col'] = df['col'] - self.ROI_COL_START

                # 鏃堕棿瀵归綈涓庨噺鍖?
                t_start = df['t_in'].min()
                df['t_bin'] = (df['t_in'] - t_start) // self.dt

                max_bin = df['t_bin'].max()
                total_frames = int(max_bin // self.T) + 1

                # 鎻愬彇鏃堕棿鍒囩墖搴忓垪
                # 璁惧畾婊戝姩姝ラ暱涓?1000 姝?(鍗?20ms 婊戝姩涓€娆?锛岃繖鏍锋埅鍙栫殑鏍锋湰鏃㈡湁閲嶅彔鍙堣兘鎻愪緵鏂颁俊鎭?
                stride = 1000
                extracted_count = 0
                max_samples_per_file = 10  # 闄愬埗姣忎釜 CSV 鏈€澶氭彁鍙?5 涓牱鏈紝闃叉鍐呭瓨鐐歌

                # 鎻愬彇鏃堕棿鍒囩墖搴忓垪 (鍔犲叆 stride 婊戝姩姝ラ暱)
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
                            coords = sparse_quantize_coordinates(locations)
                        raw_total_events += int(coords.shape[0])
                        frame_coords.append(coords)

                    sample_idx = len(sample_records)
                    sample_records.append(
                        {
                            "frame_coords": frame_coords,
                            "source_path": path,
                            "file_path": file,
                            "v_true": v_true,
                            "d_val": source_config["d_val"],
                            "K_max": source_config["K_max"],
                            "beta_max": source_config["beta_max"],
                            "log_beta_max": source_config["log_beta_max"],
                            "condition": source_config["condition"],
                            "sub_condition": source_config["sub_condition"],
                            "phantom_flag": source_config["phantom_flag"],
                            "split_group": source_config["split_group"],
                            "quality": source_config["quality"],
                            "use_for_training": source_config["use_for_training"],
                            "channel_mask_enabled": source_config["channel_mask_enabled"],
                            "channel_mask_path": source_config["channel_mask_path"],
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

                    # 杈惧埌璁惧畾鐨勬牱鏈笂闄愬悗锛屽啀璺冲嚭褰撳墠鏂囦欢鐨勮鍙?
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
                torch.zeros((1, *self.SPATIAL_SHAPE), dtype=torch.float32),
            )

            sample_extra = {
                "K_max": record["K_max"],
                "beta_max": record["beta_max"],
                "log_beta_max": record["log_beta_max"],
                "condition": record["condition"],
                "sub_condition": record["sub_condition"],
                "phantom_flag": record["phantom_flag"],
                "split_group": record["split_group"],
                "quality": record["quality"],
                "channel_mask_enabled": record["channel_mask_enabled"],
                "channel_mask_path": record["channel_mask_path"],
            }
            samples.append((sequence_data, record["v_true"], record["d_val"], env_map, source_id, sample_extra))
            self.sample_metadata.append(
                {
                    "source_path": record["source_path"],
                    "file_path": record["file_path"],
                    "v_true": record["v_true"],
                    "d_val": record["d_val"],
                    "K_max": record["K_max"],
                    "beta_max": record["beta_max"],
                    "log_beta_max": record["log_beta_max"],
                    "condition": record["condition"],
                    "sub_condition": record["sub_condition"],
                    "phantom_flag": record["phantom_flag"],
                    "split_group": record["split_group"],
                    "quality": record["quality"],
                    "use_for_training": record["use_for_training"],
                    "channel_mask_enabled": record["channel_mask_enabled"],
                    "channel_mask_path": record["channel_mask_path"],
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
        for source, summary in self.channel_mask_summary.items():
            counts = source_channel_event_counts.get(source, {"before": 0, "after": 0})
            before = int(counts["before"])
            after = int(counts["after"])
            summary["events_before_channel_mask"] = before
            if summary.get("channel_mask_enabled"):
                summary["events_after_channel_mask"] = after
                summary["channel_mask_retained_ratio"] = float(after / max(before, 1))
            else:
                summary["events_after_channel_mask"] = before
                summary["channel_mask_retained_ratio"] = 1.0
        self.event_norm_summary = self._build_event_norm_summary(
            self.sample_metadata,
            source_stats,
            source_scales,
            reference_mean,
        )
        return samples

    def get_event_norm_summary(self):
        return self.event_norm_summary

    def get_channel_mask_summary(self):
        return self.channel_mask_summary

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
    has_extra = len(batch[0]) >= 6 and isinstance(batch[0][5], dict) and "beta_max" in batch[0][5]
    metadata_idx = 6 if has_extra else 5
    has_metadata = len(batch[0]) > metadata_idx
    batched_seq_data = []

    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in batch]
        feats_t = [sample[0][t][1] for sample in batch]
        if ME is not None:
            b_coords, b_feats = ME.utils.sparse_collate(coords_t, feats_t)
        else:
            b_coords, b_feats = sparse_collate_fallback(coords_t, feats_t)
        batched_seq_data.append((b_coords, b_feats))

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    d_values = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
    env_maps = torch.stack([sample[3] for sample in batch], dim=0)
    source_ids = torch.tensor([sample[4] for sample in batch], dtype=torch.long)
    if has_extra:
        extras = [sample[5] for sample in batch]
        K_max = torch.tensor([extra.get("K_max", 1.0) for extra in extras], dtype=torch.float32)
        beta_max = torch.tensor([extra.get("beta_max", 1.0) for extra in extras], dtype=torch.float32)
        log_beta_max = torch.tensor([extra.get("log_beta_max", 0.0) for extra in extras], dtype=torch.float32)
        phantom_flag = torch.tensor([extra.get("phantom_flag", 0.0) for extra in extras], dtype=torch.float32)
        condition = [str(extra.get("condition", "unknown")) for extra in extras]
        sub_condition = [str(extra.get("sub_condition", "unknown")) for extra in extras]
        split_group = [str(extra.get("split_group", "train_val")) for extra in extras]
        quality = [str(extra.get("quality", "unknown")) for extra in extras]
    else:
        batch_size = len(batch)
        K_max = torch.ones(batch_size, dtype=torch.float32)
        beta_max = torch.ones(batch_size, dtype=torch.float32)
        log_beta_max = torch.zeros(batch_size, dtype=torch.float32)
        phantom_flag = torch.zeros(batch_size, dtype=torch.float32)
        condition = ["unknown"] * batch_size
        sub_condition = ["unknown"] * batch_size
        split_group = ["train_val"] * batch_size
        quality = ["legacy"] * batch_size

    if has_metadata:
        metadata = [sample[metadata_idx] for sample in batch]
        return (
            batched_seq_data,
            labels,
            d_values,
            env_maps,
            source_ids,
            K_max,
            beta_max,
            log_beta_max,
            condition,
            sub_condition,
            split_group,
            quality,
            phantom_flag,
            metadata,
        )
    return (
        batched_seq_data,
        labels,
        d_values,
        env_maps,
        source_ids,
        K_max,
        beta_max,
        log_beta_max,
        condition,
        sub_condition,
        split_group,
        quality,
        phantom_flag,
    )
