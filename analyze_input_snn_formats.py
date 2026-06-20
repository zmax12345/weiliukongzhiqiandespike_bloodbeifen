import csv
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

try:
    from tqdm.auto import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable, total=None, desc=None, leave=False, dynamic_ncols=True):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

    def tqdm(iterable, total=None, desc=None, leave=False, dynamic_ncols=True):
        return _TqdmFallback(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)

from dataset_robust import FlexibleBloodFlowDataset, sequence_sparse_collate
from dense_block_manager import DenseBlockManager
from snn_encoder import SNNEncoder


dt_us = 20
snn_bin_size_list = [1, 5, 10, 15, 20, 40, 60, 100, 150, 250]
window_ms_list = [50, 100, 150, 200, 300]
batch_size = 2
max_batches_per_split = 20
max_velocity = 2.0
spatial_shape = (100, 368)
patch_shape = (50, 46)
event_norm_mode = "source_scale"
event_norm_clip = (0.25, 4.0)
event_intensity_jitter_range = None
snn_input_scale_mode = "sqrt"

train_env_config = {
    "/data/zm/Moshaboli/new_data/no1": 0.018938,
    "/data/zm/Moshaboli/new_data/no4": 0.01973,
    "/data/zm/Moshaboli/new_data/no2": 0.01942,
}

val_env_config = {
    "/data/zm/Moshaboli/new_data/no3": 0.01963,
}

mask_path = "/data/zm/Moshaboli/new_data/other_data/3.0_mask (2)_hot_pixel_mask.npy"
report_dir = "/data/zm/Moshaboli/new_data/Markdown"

RAW_STAT_NAMES = [
    "total_events",
    "mean_count",
    "diff_mean",
    "diff_std",
    "fano",
    "acf_lag1",
    "acf_lag5",
    "growth_ratio",
]

LAYER_DIAG_NAMES = [
    "conv_out_max",
    "mem_max",
    "mthr_max",
    "beta_mean",
    "b_mean",
    "kernel_norm_mean",
]

CSV_FIELDS = [
    "split",
    "window_ms",
    "base_total_steps",
    "usable_base_steps",
    "dropped_base_steps",
    "snn_bin_size",
    "snn_step_us",
    "snn_steps",
    "input_scale_mode",
    "samples_used",
    "batches_used",
    "input_step_count_mean",
    "input_step_count_std",
    "input_step_count_min",
    "input_step_count_max",
    "nonzero_step_ratio",
    "active_pixel_ratio",
    "sample_total_mean",
    "sample_total_std",
    "total_events_corr",
    "diff_mean_corr",
    "fano_corr",
    "acf_lag1_corr",
    "best_raw_stat_name",
    "best_raw_stat_abs_corr",
    "raw_stats_linear_mae",
    "raw_stats_linear_corr",
    "layer1_spike_rate",
    "layer2_spike_rate",
    "layer3_spike_rate",
    "layer1_conv_out_max",
    "layer2_conv_out_max",
    "layer3_conv_out_max",
    "layer1_mem_max",
    "layer2_mem_max",
    "layer3_mem_max",
    "layer1_mthr_max",
    "layer2_mthr_max",
    "layer3_mthr_max",
    "layer1_beta_mean",
    "layer2_beta_mean",
    "layer3_beta_mean",
    "layer1_b_mean",
    "layer2_b_mean",
    "layer3_b_mean",
    "layer1_kernel_norm_mean",
    "layer2_kernel_norm_mean",
    "layer3_kernel_norm_mean",
    "feat1_mean",
    "feat1_std",
    "feat2_mean",
    "feat2_std",
    "feat3_mean",
    "feat3_std",
    "feat1_min",
    "feat2_min",
    "feat3_min",
    "feat1_max",
    "feat2_max",
    "feat3_max",
    "silent_fraction_1",
    "silent_fraction_2",
    "silent_fraction_3",
    "snn_feature_global_std",
    "snn_feature_best_abs_corr",
    "snn_feature_linear_mae",
    "snn_feature_linear_corr",
    "status",
    "recommendation",
]


class SourceVelocityBatchSampler(Sampler):
    def __init__(self, source_velocity_sample_indices, plan, seed):
        self.source_velocity_sample_indices = source_velocity_sample_indices
        self.plan = plan
        self.seed = seed
        self.sources = list(plan["sources"])
        self.velocities = list(plan["velocities"])

    def __len__(self):
        return self.plan["effective_batches"]

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        batches = []
        passes_per_source = self.plan["passes_per_source"]
        velocities_per_batch = self.plan["velocities_per_batch"]
        for source in self.sources:
            velocity_map = self.source_velocity_sample_indices[source]
            shuffled_indices = {}
            for velocity in self.velocities:
                indices = np.array(velocity_map[velocity], dtype=np.int64)
                shuffled_indices[velocity] = rng.permutation(indices).tolist()
            for pass_idx in range(passes_per_source):
                velocity_order = rng.permutation(self.velocities).tolist()
                for start_idx in range(0, len(velocity_order), velocities_per_batch):
                    velocity_chunk = velocity_order[start_idx:start_idx + velocities_per_batch]
                    batch = [shuffled_indices[velocity][pass_idx] for velocity in velocity_chunk]
                    rng.shuffle(batch)
                    batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)


def compute_source_velocity_sampling_plan(dataset, batch_size, max_batches, split_name):
    source_velocity_indices = getattr(dataset, "source_velocity_sample_indices", {})
    if not source_velocity_indices:
        raise ValueError(f"{split_name} dataset has no source/velocity indices for velocity-cycled batches.")

    sources = sorted(source_velocity_indices.keys())
    velocity_sets = [set(source_velocity_indices[source].keys()) for source in sources]
    common_velocities = sorted(set.intersection(*velocity_sets))
    if not common_velocities:
        raise ValueError(f"{split_name} dataset has no common velocities across sources.")
    if batch_size > len(common_velocities) or len(common_velocities) % batch_size != 0:
        raise ValueError(
            f"{split_name} batch_size must divide the number of common velocities. "
            f"Got batch_size={batch_size}, common_velocities={len(common_velocities)}."
        )

    passes_per_source_available = min(
        min(len(source_velocity_indices[source][velocity]) for velocity in common_velocities)
        for source in sources
    )
    velocity_chunks_per_source = len(common_velocities) // batch_size
    batch_balance_unit = len(sources) * velocity_chunks_per_source
    max_balanced_batches = passes_per_source_available * batch_balance_unit
    if max_batches is None:
        effective_batches = max_balanced_batches
    else:
        effective_batches = min(max_batches, max_balanced_batches)
        effective_batches -= effective_batches % batch_balance_unit
    if effective_batches <= 0:
        raise ValueError(
            f"{split_name} max_batches={max_batches} is too small for source-balanced velocity batches."
        )
    batches_per_source = effective_batches // len(sources)
    passes_per_source = batches_per_source // velocity_chunks_per_source
    return {
        "sampling_mode": "source_velocity_cycle",
        "requested_batches": max_batches,
        "effective_batches": effective_batches,
        "num_sources": len(sources),
        "num_velocities": len(common_velocities),
        "velocities_per_batch": batch_size,
        "velocity_chunks_per_source": velocity_chunks_per_source,
        "velocities": common_velocities,
        "sources": sources,
        "batches_per_source": batches_per_source,
        "passes_per_source": passes_per_source,
        "samples_per_source": batches_per_source * batch_size,
        "was_adjusted": max_batches is not None and effective_batches != max_batches,
    }


def build_source_velocity_loader(dataset, batch_size, num_workers, collate_fn, max_batches, split_name, seed):
    plan = compute_source_velocity_sampling_plan(dataset, batch_size, max_batches, split_name)
    sampler = SourceVelocityBatchSampler(dataset.source_velocity_sample_indices, plan, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers), plan


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update_tensor(self, tensor):
        values = tensor.detach()
        self.count += values.numel()
        self.sum += float(values.sum().cpu())
        self.sumsq += float(values.pow(2).sum().cpu())
        self.min = min(self.min, float(values.min().cpu()))
        self.max = max(self.max, float(values.max().cpu()))

    def update_array(self, array):
        values = np.asarray(array, dtype=np.float64)
        if values.size == 0:
            return
        self.count += int(values.size)
        self.sum += float(values.sum())
        self.sumsq += float(np.square(values).sum())
        self.min = min(self.min, float(values.min()))
        self.max = max(self.max, float(values.max()))

    @property
    def mean(self):
        if self.count == 0:
            return float("nan")
        return self.sum / self.count

    @property
    def std(self):
        if self.count == 0:
            return float("nan")
        var = max(self.sumsq / self.count - self.mean ** 2, 0.0)
        return math.sqrt(var)

    def as_min(self):
        return self.min if self.count else float("nan")

    def as_max(self):
        return self.max if self.count else float("nan")


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or y.size < 2:
        return float("nan")
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def linear_baseline(features, targets, ridge=1e-3, seed=20260511):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] < 4:
        return float("nan"), float("nan")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.shape[0] < 4:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    folds = np.array_split(indices, min(5, x.shape[0]))
    pred = np.full_like(y, fill_value=np.nan, dtype=np.float64)

    for val_idx in folds:
        if val_idx.size == 0:
            continue
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[val_idx] = False
        if train_mask.sum() < 2:
            continue

        x_train = x[train_mask]
        y_train = y[train_mask]
        x_val = x[val_idx]
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
        x_val = np.concatenate([x_val, np.ones((x_val.shape[0], 1))], axis=1)

        eye = np.eye(x_train.shape[1])
        eye[-1, -1] = 0.0
        lhs = x_train.T @ x_train + ridge * eye
        rhs = x_train.T @ y_train
        try:
            weight = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weight = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        pred[val_idx] = x_val @ weight

    finite_pred = np.isfinite(pred)
    if not np.any(finite_pred):
        return float("nan"), float("nan")
    mae = float(np.mean(np.abs(pred[finite_pred] - y[finite_pred])))
    corr = safe_pearson(pred[finite_pred], y[finite_pred])
    return mae, corr


def apply_input_scale(x_snn, snn_bin_size, mode):
    if mode == "none":
        return x_snn
    if mode == "mean":
        return x_snn / float(snn_bin_size)
    if mode == "sqrt":
        return x_snn / math.sqrt(float(snn_bin_size))
    raise ValueError(f"Unsupported snn_input_scale_mode={mode}")


def dense_base_block_to_snn_steps(manager, base_block_idx, base_block_size, snn_bin_size, device):
    x_base = manager.get_block_dense(base_block_idx, base_block_size).to(device)
    if x_base.shape[0] % snn_bin_size != 0:
        raise ValueError("base_block_size must be divisible by snn_bin_size.")
    num_snn_steps = x_base.shape[0] // snn_bin_size
    x_snn = x_base.reshape(
        num_snn_steps,
        snn_bin_size,
        x_base.shape[1],
        x_base.shape[2],
        x_base.shape[3],
        x_base.shape[4],
    ).sum(dim=1)
    return apply_input_scale(x_snn, snn_bin_size, snn_input_scale_mode)


def compute_raw_temporal_stats(counts):
    eps = 1e-6
    counts = counts.detach().to(torch.float32)
    total_events = counts.sum(dim=0)
    mean_count = counts.mean(dim=0)
    if counts.shape[0] > 1:
        diff = torch.abs(counts[1:] - counts[:-1])
        diff_mean = diff.mean(dim=0)
        diff_std = diff.std(dim=0, unbiased=False)
    else:
        diff_mean = counts.new_zeros(counts.shape[1])
        diff_std = counts.new_zeros(counts.shape[1])
    fano = counts.var(dim=0, unbiased=False) / (mean_count + eps)
    acf_lag1 = acf_lag(counts, 1, eps)
    acf_lag5 = acf_lag(counts, 5, eps)
    half = counts.shape[0] // 2
    first_half = counts[:half].sum(dim=0) if half > 0 else counts.new_zeros(counts.shape[1])
    second_half = counts[half:].sum(dim=0)
    growth_ratio = torch.log1p(second_half) - torch.log1p(first_half)
    return torch.stack(
        [
            torch.log1p(total_events),
            torch.log1p(mean_count),
            torch.log1p(diff_mean),
            torch.log1p(diff_std),
            torch.clamp(fano, 0.0, 100.0),
            acf_lag1,
            acf_lag5,
            torch.clamp(growth_ratio, -10.0, 10.0),
        ],
        dim=1,
    )


def acf_lag(counts, lag, eps):
    if counts.shape[0] <= lag:
        return counts.new_zeros(counts.shape[1])
    centered = counts - counts.mean(dim=0, keepdim=True)
    numerator = (centered[:-lag] * centered[lag:]).mean(dim=0)
    denom = counts.var(dim=0, unbiased=False) + eps
    return torch.clamp(numerator / denom, -1.0, 1.0)


def append_snn_sample_features(sample_features, feat_accum, num_snn_steps, batch_size, patches_per_sample):
    pieces = []
    for accum in feat_accum:
        feat = accum / float(num_snn_steps)
        patch_feat = feat.mean(dim=(2, 3))
        patch_feat = patch_feat.view(batch_size, patches_per_sample, patch_feat.shape[1])
        pieces.append(patch_feat.mean(dim=1))
        pieces.append(patch_feat.std(dim=1, unbiased=False))
    sample_features.append(torch.cat(pieces, dim=1).detach().cpu().numpy())


def analyze_dataset_split(
    split_name,
    dataset,
    base_total_steps,
    snn_bin_size,
    device,
    max_batches,
):
    usable_base_steps = (base_total_steps // snn_bin_size) * snn_bin_size
    dropped_base_steps = base_total_steps - usable_base_steps
    snn_steps = usable_base_steps // snn_bin_size
    row_base = {
        "split": split_name,
        "window_ms": int(base_total_steps * dt_us / 1000),
        "base_total_steps": base_total_steps,
        "usable_base_steps": usable_base_steps,
        "dropped_base_steps": dropped_base_steps,
        "snn_bin_size": snn_bin_size,
        "snn_step_us": snn_bin_size * dt_us,
        "snn_steps": snn_steps,
        "input_scale_mode": snn_input_scale_mode,
    }
    if len(dataset) == 0 or snn_steps == 0:
        return {
            **row_base,
            **empty_metrics(),
            "samples_used": 0,
            "batches_used": 0,
            "status": "no_samples",
            "recommendation": "Increase window/data availability or skip this setting.",
        }, []

    try:
        loader, plan = build_source_velocity_loader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=sequence_sparse_collate,
            max_batches=max_batches,
            split_name=split_name,
            seed=20260511 + base_total_steps + snn_bin_size,
        )
    except ValueError:
        return {
            **row_base,
            **empty_metrics(),
            "samples_used": 0,
            "batches_used": 0,
            "status": "no_samples",
            "recommendation": "Increase window/data availability or skip this setting.",
        }, []

    snn = SNNEncoder(in_channels=1).to(device)
    snn.eval()

    input_step_stats = RunningStats()
    sample_total_stats = RunningStats()
    feat_stats = [RunningStats(), RunningStats(), RunningStats()]
    layer_diag_stats = [
        {diag_name: RunningStats() for diag_name in LAYER_DIAG_NAMES}
        for _ in range(3)
    ]
    spike_sum = [0.0, 0.0, 0.0]
    spike_numel = [0, 0, 0]
    silent_count = [0, 0, 0]
    silent_numel = [0, 0, 0]
    active_pixel_count = 0
    active_pixel_total = 0
    nonzero_step_count = 0
    nonzero_step_total = 0

    raw_stats_list = []
    sample_feature_list = []
    y_list = []
    per_velocity_rows = []
    processed_batches = 0

    base_block_size = snn_bin_size * max(1, 100 // snn_bin_size)
    base_blocks = usable_base_steps // base_block_size
    tail_snn_steps = (usable_base_steps - base_blocks * base_block_size) // snn_bin_size

    progress = tqdm(
        enumerate(loader),
        total=plan["effective_batches"],
        desc=f"{split_name} window={row_base['window_ms']}ms K={snn_bin_size}",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch_idx, (x_seq_sparse_data, y_true, d_values, env_maps, source_ids) in progress:
            if max_batches is not None and processed_batches >= plan["effective_batches"]:
                break
            manager = DenseBlockManager(
                x_seq_sparse_data,
                batch_size=y_true.shape[0],
                spatial_shape=spatial_shape,
                patch_shape=patch_shape,
            )
            patches_per_sample = manager.patches_per_sample
            patch_batch_size = y_true.shape[0] * patches_per_sample
            mems = snn.init_state(patch_batch_size, patch_shape, device)
            feat_accum = [torch.zeros_like(mem) for mem in mems]
            step_count_chunks = []

            def process_x_snn(x_snn):
                nonlocal mems, active_pixel_count, active_pixel_total, nonzero_step_count, nonzero_step_total
                step_event_count = x_snn.sum(dim=(2, 3, 4))
                step_event_count_sample = step_event_count.view(
                    x_snn.shape[0],
                    y_true.shape[0],
                    patches_per_sample,
                ).sum(dim=2)
                step_count_chunks.append(step_event_count_sample.detach())
                input_step_stats.update_tensor(step_event_count_sample)
                nonzero_step_count += int((step_event_count_sample > 0).sum().item())
                nonzero_step_total += step_event_count_sample.numel()
                active_pixel_count += int((x_snn > 0).sum().item())
                active_pixel_total += x_snn.numel()

                for t in range(x_snn.shape[0]):
                    spikes, mems = snn.forward_step(x_snn[t], mems)
                    for layer_idx, spike_tensor in enumerate(spikes):
                        spike_sum[layer_idx] += float(spike_tensor.sum().item())
                        spike_numel[layer_idx] += spike_tensor.numel()
                        feat_accum[layer_idx] += spike_tensor
                    for layer_idx, diag in enumerate(snn.get_layer_diagnostics()):
                        for diag_name in LAYER_DIAG_NAMES:
                            diag_value = diag.get(diag_name)
                            if diag_value is not None and np.isfinite(diag_value):
                                layer_diag_stats[layer_idx][diag_name].update_array([diag_value])

            for base_block_idx in range(base_blocks):
                x_snn = dense_base_block_to_snn_steps(
                    manager,
                    base_block_idx,
                    base_block_size,
                    snn_bin_size,
                    device,
                )
                process_x_snn(x_snn)

            first_tail_step = base_blocks * (base_block_size // snn_bin_size)
            for tail_idx in range(tail_snn_steps):
                x_snn = dense_base_block_to_snn_steps(
                    manager,
                    first_tail_step + tail_idx,
                    snn_bin_size,
                    snn_bin_size,
                    device,
                )
                process_x_snn(x_snn)

            counts = torch.cat(step_count_chunks, dim=0)
            sample_total_stats.update_tensor(counts.sum(dim=0))
            raw_stats_list.append(compute_raw_temporal_stats(counts).detach().cpu().numpy())
            append_snn_sample_features(sample_feature_list, feat_accum, snn_steps, y_true.shape[0], patches_per_sample)
            y_list.append(y_true.detach().cpu().numpy())

            for layer_idx, accum in enumerate(feat_accum):
                feat = accum / float(snn_steps)
                feat_stats[layer_idx].update_tensor(feat)
                silent_count[layer_idx] += int((feat == 0).sum().item())
                silent_numel[layer_idx] += feat.numel()

            processed_batches += 1
            progress.set_postfix(
                l1=f"{spike_sum[0] / max(spike_numel[0], 1):.2e}",
                l2=f"{spike_sum[1] / max(spike_numel[1], 1):.2e}",
                l3=f"{spike_sum[2] / max(spike_numel[2], 1):.2e}",
            )
    progress.close()

    if processed_batches == 0:
        return {
            **row_base,
            **empty_metrics(),
            "samples_used": 0,
            "batches_used": 0,
            "status": "no_samples",
            "recommendation": "Increase window/data availability or skip this setting.",
        }, []

    raw_stats = np.concatenate(raw_stats_list, axis=0)
    sample_features = np.concatenate(sample_feature_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    raw_corrs = {name: safe_pearson(raw_stats[:, idx], y) for idx, name in enumerate(RAW_STAT_NAMES)}
    raw_abs_corrs = {
        name: abs(value)
        for name, value in raw_corrs.items()
        if np.isfinite(value)
    }
    if raw_abs_corrs:
        best_raw_stat_name = max(raw_abs_corrs, key=raw_abs_corrs.get)
        best_raw_stat_abs_corr = raw_abs_corrs[best_raw_stat_name]
    else:
        best_raw_stat_name = ""
        best_raw_stat_abs_corr = float("nan")
    raw_linear_mae, raw_linear_corr = linear_baseline(raw_stats, y)

    snn_feature_corrs = [safe_pearson(sample_features[:, idx], y) for idx in range(sample_features.shape[1])]
    finite_feature_corrs = [abs(value) for value in snn_feature_corrs if np.isfinite(value)]
    snn_feature_best_abs_corr = max(finite_feature_corrs) if finite_feature_corrs else float("nan")
    snn_feature_linear_mae, snn_feature_linear_corr = linear_baseline(sample_features, y)

    layer_spike_rates = [
        spike_sum[idx] / max(spike_numel[idx], 1)
        for idx in range(3)
    ]
    row = {
        **row_base,
        "samples_used": int(y.shape[0]),
        "batches_used": processed_batches,
        "input_step_count_mean": input_step_stats.mean,
        "input_step_count_std": input_step_stats.std,
        "input_step_count_min": input_step_stats.as_min(),
        "input_step_count_max": input_step_stats.as_max(),
        "nonzero_step_ratio": nonzero_step_count / max(nonzero_step_total, 1),
        "active_pixel_ratio": active_pixel_count / max(active_pixel_total, 1),
        "sample_total_mean": sample_total_stats.mean,
        "sample_total_std": sample_total_stats.std,
        "total_events_corr": raw_corrs["total_events"],
        "diff_mean_corr": raw_corrs["diff_mean"],
        "fano_corr": raw_corrs["fano"],
        "acf_lag1_corr": raw_corrs["acf_lag1"],
        "best_raw_stat_name": best_raw_stat_name,
        "best_raw_stat_abs_corr": best_raw_stat_abs_corr,
        "raw_stats_linear_mae": raw_linear_mae,
        "raw_stats_linear_corr": raw_linear_corr,
        "layer1_spike_rate": layer_spike_rates[0],
        "layer2_spike_rate": layer_spike_rates[1],
        "layer3_spike_rate": layer_spike_rates[2],
        "layer1_conv_out_max": layer_diag_stats[0]["conv_out_max"].as_max(),
        "layer2_conv_out_max": layer_diag_stats[1]["conv_out_max"].as_max(),
        "layer3_conv_out_max": layer_diag_stats[2]["conv_out_max"].as_max(),
        "layer1_mem_max": layer_diag_stats[0]["mem_max"].as_max(),
        "layer2_mem_max": layer_diag_stats[1]["mem_max"].as_max(),
        "layer3_mem_max": layer_diag_stats[2]["mem_max"].as_max(),
        "layer1_mthr_max": layer_diag_stats[0]["mthr_max"].as_max(),
        "layer2_mthr_max": layer_diag_stats[1]["mthr_max"].as_max(),
        "layer3_mthr_max": layer_diag_stats[2]["mthr_max"].as_max(),
        "layer1_beta_mean": layer_diag_stats[0]["beta_mean"].mean,
        "layer2_beta_mean": layer_diag_stats[1]["beta_mean"].mean,
        "layer3_beta_mean": layer_diag_stats[2]["beta_mean"].mean,
        "layer1_b_mean": layer_diag_stats[0]["b_mean"].mean,
        "layer2_b_mean": layer_diag_stats[1]["b_mean"].mean,
        "layer3_b_mean": layer_diag_stats[2]["b_mean"].mean,
        "layer1_kernel_norm_mean": layer_diag_stats[0]["kernel_norm_mean"].mean,
        "layer2_kernel_norm_mean": layer_diag_stats[1]["kernel_norm_mean"].mean,
        "layer3_kernel_norm_mean": layer_diag_stats[2]["kernel_norm_mean"].mean,
        "feat1_mean": feat_stats[0].mean,
        "feat1_std": feat_stats[0].std,
        "feat2_mean": feat_stats[1].mean,
        "feat2_std": feat_stats[1].std,
        "feat3_mean": feat_stats[2].mean,
        "feat3_std": feat_stats[2].std,
        "feat1_min": feat_stats[0].as_min(),
        "feat2_min": feat_stats[1].as_min(),
        "feat3_min": feat_stats[2].as_min(),
        "feat1_max": feat_stats[0].as_max(),
        "feat2_max": feat_stats[1].as_max(),
        "feat3_max": feat_stats[2].as_max(),
        "silent_fraction_1": silent_count[0] / max(silent_numel[0], 1),
        "silent_fraction_2": silent_count[1] / max(silent_numel[1], 1),
        "silent_fraction_3": silent_count[2] / max(silent_numel[2], 1),
        "snn_feature_global_std": float(np.nanstd(sample_features)),
        "snn_feature_best_abs_corr": snn_feature_best_abs_corr,
        "snn_feature_linear_mae": snn_feature_linear_mae,
        "snn_feature_linear_corr": snn_feature_linear_corr,
    }
    status, recommendation = make_recommendation(row)
    row["status"] = status
    row["recommendation"] = recommendation

    for velocity in sorted(set(y.tolist())):
        mask = y == velocity
        per_velocity_rows.append(
            {
                "split": split_name,
                "window_ms": row["window_ms"],
                "snn_bin_size": snn_bin_size,
                "velocity": float(velocity),
                "samples": int(mask.sum()),
                "raw_total_events_mean": float(np.mean(raw_stats[mask, RAW_STAT_NAMES.index("total_events")])),
                "snn_feature_norm_mean": float(np.mean(np.linalg.norm(sample_features[mask], axis=1))),
            }
        )
    return sanitize_row(row), per_velocity_rows


def empty_metrics():
    metrics = {}
    for field in CSV_FIELDS:
        if field in {
            "split",
            "window_ms",
            "base_total_steps",
            "usable_base_steps",
            "dropped_base_steps",
            "snn_bin_size",
            "snn_step_us",
            "snn_steps",
            "input_scale_mode",
            "samples_used",
            "batches_used",
            "status",
            "recommendation",
        }:
            continue
        metrics[field] = float("nan")
    metrics["best_raw_stat_name"] = ""
    return metrics


def make_recommendation(row):
    if row["samples_used"] == 0:
        return "no_samples", "Increase window/data availability or skip this setting."
    if row["layer1_spike_rate"] < 1e-8:
        return "dead_snn_layer1", "Input too sparse or threshold too high. Try larger snn_bin_size or lower threshold/input gain."
    if row["layer1_spike_rate"] > 0 and row["layer2_spike_rate"] < 1e-8:
        return "dead_snn_layer2", "First layer fires but deeper layer is silent. Try lowering deeper thresholds or increasing feature gain."
    if row["feat3_std"] < 1e-8 and row["layer3_spike_rate"] > 0:
        return "low_feature_variance", "SNN fires but features are nearly constant. Check saturation or normalization."
    if np.isfinite(row["snn_feature_linear_mae"]) and row["snn_feature_linear_mae"] < 0.5:
        return "promising_snn_format", "This input format preserves velocity information in SNN features."
    if (
        np.isfinite(row["raw_stats_linear_mae"])
        and row["raw_stats_linear_mae"] < 0.5
        and (not np.isfinite(row["snn_feature_linear_mae"]) or row["snn_feature_linear_mae"] >= 0.5)
    ):
        return "raw_signal_present_snn_lost", "Raw temporal signal exists, but SNN encoding loses it. Adjust SNN neuron/input scale."
    return "inconclusive", "Needs more batches or parameter tuning."


def sanitize_row(row):
    clean = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, float)):
            value = float(value)
            if not np.isfinite(value):
                clean[key] = float("nan")
            else:
                clean[key] = value
        else:
            clean[key] = value
    return clean


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def format_float(value, digits=6):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_markdown(path, rows, per_velocity_rows, csv_path):
    sorted_rows = sorted(rows, key=lambda r: (r["split"], r["window_ms"], r["snn_bin_size"]))
    promising = sorted(
        [r for r in sorted_rows if r["status"] == "promising_snn_format"],
        key=lambda r: r["snn_feature_linear_mae"],
    )
    raw_lost = [
        r for r in sorted_rows
        if r["status"] == "raw_signal_present_snn_lost"
    ]
    dead = [
        r for r in sorted_rows
        if r["status"] in {"dead_snn_layer1", "dead_snn_layer2"}
    ]

    lines = [
        "# Input SNN Format Analysis",
        "",
        "## Run Config",
        "",
        f"- dt_us: `{dt_us}`",
        f"- snn_bin_size_list: `{snn_bin_size_list}`",
        f"- window_ms_list: `{window_ms_list}`",
        f"- input_scale_mode: `{snn_input_scale_mode}`",
        f"- max_batches_per_split: `{max_batches_per_split}`",
        f"- event_norm_mode: `{event_norm_mode}`",
        f"- event_norm_clip: `{event_norm_clip}`",
        f"- batch_size: `{batch_size}`",
        f"- max_velocity: `{max_velocity}`",
        f"- spatial_shape: `{spatial_shape}`",
        f"- patch_shape: `{patch_shape}`",
        f"- csv_path: `{csv_path}`",
        "",
        "### Train Sources",
        "",
    ]
    for path_key, d_value in train_env_config.items():
        lines.append(f"- `{path_key}` -> d=`{d_value}`")
    lines.extend(["", "### Val Sources", ""])
    for path_key, d_value in val_env_config.items():
        lines.append(f"- `{path_key}` -> d=`{d_value}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Split | Window ms | K | Step us | Samples | Input Mean | Nonzero Step | L1 Rate | L2 Rate | L3 Rate | Feat3 Std | Raw MAE | SNN MAE | Best Raw Corr | Best SNN Corr | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted_rows:
        lines.append(
            f"| {row['split']} | {row['window_ms']} | {row['snn_bin_size']} | {row['snn_step_us']} | "
            f"{row['samples_used']} | {format_float(row['input_step_count_mean'])} | "
            f"{format_float(row['nonzero_step_ratio'])} | {format_float(row['layer1_spike_rate'], 8)} | "
            f"{format_float(row['layer2_spike_rate'], 8)} | {format_float(row['layer3_spike_rate'], 8)} | "
            f"{format_float(row['feat3_std'], 8)} | {format_float(row['raw_stats_linear_mae'])} | "
            f"{format_float(row['snn_feature_linear_mae'])} | {format_float(row['best_raw_stat_abs_corr'])} | "
            f"{format_float(row['snn_feature_best_abs_corr'])} | `{row['status']}` |"
        )

    lines.extend(
        [
            "",
            "## Legacy Neuron Diagnostics",
            "",
            "| Split | Window ms | K | L1 Conv Max | L1 Mem Max | L1 Mthr Max | L1 b | L1 Norm | L2 Mthr Max | L2 b | L2 Norm | L3 Mthr Max | L3 b | L3 Norm |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted_rows:
        lines.append(
            f"| {row['split']} | {row['window_ms']} | {row['snn_bin_size']} | "
            f"{format_float(row['layer1_conv_out_max'])} | {format_float(row['layer1_mem_max'])} | "
            f"{format_float(row['layer1_mthr_max'])} | {format_float(row['layer1_b_mean'])} | "
            f"{format_float(row['layer1_kernel_norm_mean'])} | {format_float(row['layer2_mthr_max'])} | "
            f"{format_float(row['layer2_b_mean'])} | {format_float(row['layer2_kernel_norm_mean'])} | "
            f"{format_float(row['layer3_mthr_max'])} | {format_float(row['layer3_b_mean'])} | "
            f"{format_float(row['layer3_kernel_norm_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Recommended Candidates",
            "",
            "| Split | Window ms | K | Step us | SNN MAE | SNN Corr | L1 Rate | L2 Rate | L3 Rate | Recommendation |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if promising:
        for row in promising:
            lines.append(
                f"| {row['split']} | {row['window_ms']} | {row['snn_bin_size']} | {row['snn_step_us']} | "
                f"{format_float(row['snn_feature_linear_mae'])} | {format_float(row['snn_feature_linear_corr'])} | "
                f"{format_float(row['layer1_spike_rate'], 8)} | {format_float(row['layer2_spike_rate'], 8)} | "
                f"{format_float(row['layer3_spike_rate'], 8)} | {row['recommendation']} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | No promising format found. |")

    lines.extend(
        [
            "",
            "## Dead SNN Formats",
            "",
            "| Split | Window ms | K | Step us | Status | L1 Rate | L2 Rate | L3 Rate | Recommendation |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    if dead:
        for row in dead:
            lines.append(
                f"| {row['split']} | {row['window_ms']} | {row['snn_bin_size']} | {row['snn_step_us']} | "
                f"`{row['status']}` | {format_float(row['layer1_spike_rate'], 8)} | "
                f"{format_float(row['layer2_spike_rate'], 8)} | {format_float(row['layer3_spike_rate'], 8)} | "
                f"{row['recommendation']} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | No dead formats found. |")

    lines.extend(
        [
            "",
            "## Raw Signal Present But SNN Lost",
            "",
            "| Split | Window ms | K | Raw MAE | SNN MAE | Best Raw Stat | Best Raw Corr | Recommendation |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    if raw_lost:
        for row in raw_lost:
            lines.append(
                f"| {row['split']} | {row['window_ms']} | {row['snn_bin_size']} | "
                f"{format_float(row['raw_stats_linear_mae'])} | {format_float(row['snn_feature_linear_mae'])} | "
                f"`{row['best_raw_stat_name']}` | {format_float(row['best_raw_stat_abs_corr'])} | "
                f"{row['recommendation']} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | No raw-present/SNN-lost formats found. |")

    lines.extend(
        [
            "",
            "## Per Velocity Detail",
            "",
            "| Split | Window ms | K | Velocity | Samples | Raw Total Events Mean | SNN Feature Norm Mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_velocity_rows:
        lines.append(
            f"| {row['split']} | {row['window_ms']} | {row['snn_bin_size']} | "
            f"{format_float(row['velocity'])} | {row['samples']} | "
            f"{format_float(row['raw_total_events_mean'])} | {format_float(row['snn_feature_norm_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- Check whether K=1 is silent in layer1.",
            "- Find the smallest K where layer1 begins firing, then whether layer2/layer3 also fire.",
            "- If raw stats have low MAE but SNN feature MAE is high, the SNN encoding is losing information.",
            "- Prefer formats with nonzero layer3 activity, nonzero feature std, and low SNN linear MAE.",
            "- Use the recommended candidates as the next `snn_bin_size` and `window_ms` for the formal SNN-CNN training run.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, f"input_snn_format_analysis_{timestamp}.csv")
    md_path = os.path.join(report_dir, f"input_snn_format_analysis_{timestamp}.md")

    all_rows = []
    all_per_velocity_rows = []

    for window_ms in window_ms_list:
        base_total_steps = int(window_ms * 1000 / dt_us)
        print(f"\n===== Build datasets for window={window_ms}ms ({base_total_steps} base steps) =====")
        train_ds = FlexibleBloodFlowDataset(
            train_env_config,
            mask_path=mask_path,
            T=1,
            seq_len=base_total_steps,
            dt_us=dt_us,
            max_velocity=max_velocity,
            event_norm_mode=event_norm_mode,
            event_norm_reference_mean=None,
            event_norm_clip=event_norm_clip,
            event_intensity_jitter_range=event_intensity_jitter_range,
        )
        train_event_norm_stats = train_ds.get_reference_event_norm_stats() if len(train_ds) > 0 else None
        val_ds = FlexibleBloodFlowDataset(
            val_env_config,
            mask_path=mask_path,
            T=1,
            seq_len=base_total_steps,
            dt_us=dt_us,
            max_velocity=max_velocity,
            event_norm_mode=event_norm_mode,
            event_norm_stats=train_event_norm_stats,
            event_norm_reference_mean=(
                train_event_norm_stats["reference_mean_events_per_sample"]
                if train_event_norm_stats is not None
                else None
            ),
            event_norm_clip=event_norm_clip,
            event_intensity_jitter_range=event_intensity_jitter_range,
        )

        for snn_bin_size in snn_bin_size_list:
            for split_name, dataset in (("train", train_ds), ("val", val_ds)):
                print(f"Analyze split={split_name}, window={window_ms}ms, K={snn_bin_size}")
                row, per_velocity_rows = analyze_dataset_split(
                    split_name=split_name,
                    dataset=dataset,
                    base_total_steps=base_total_steps,
                    snn_bin_size=snn_bin_size,
                    device=device,
                    max_batches=max_batches_per_split,
                )
                all_rows.append(row)
                all_per_velocity_rows.extend(per_velocity_rows)
                write_csv(csv_path, all_rows)
                write_markdown(md_path, all_rows, all_per_velocity_rows, csv_path)

    write_csv(csv_path, all_rows)
    write_markdown(md_path, all_rows, all_per_velocity_rows, csv_path)

    promising = sorted(
        [r for r in all_rows if r["status"] == "promising_snn_format"],
        key=lambda r: r["snn_feature_linear_mae"],
    )
    raw_best = sorted(
        [r for r in all_rows if np.isfinite(r.get("raw_stats_linear_mae", float("nan")))],
        key=lambda r: r["raw_stats_linear_mae"],
    )
    dead = [r for r in all_rows if r["status"] in {"dead_snn_layer1", "dead_snn_layer2"}]

    print("\nBest promising formats by snn_feature_linear_mae")
    for row in promising[:10]:
        print(
            f"  {row['split']} window={row['window_ms']}ms K={row['snn_bin_size']} "
            f"snn_mae={format_float(row['snn_feature_linear_mae'])} "
            f"rates=({format_float(row['layer1_spike_rate'], 8)}, "
            f"{format_float(row['layer2_spike_rate'], 8)}, {format_float(row['layer3_spike_rate'], 8)})"
        )
    if not promising:
        print("  none")

    print("\nBest raw formats by raw_stats_linear_mae")
    for row in raw_best[:10]:
        print(
            f"  {row['split']} window={row['window_ms']}ms K={row['snn_bin_size']} "
            f"raw_mae={format_float(row['raw_stats_linear_mae'])} "
            f"best_raw={row['best_raw_stat_name']} corr={format_float(row['best_raw_stat_abs_corr'])}"
        )
    if not raw_best:
        print("  none")

    print("\nDead SNN formats")
    for row in dead[:20]:
        print(f"  {row['split']} window={row['window_ms']}ms K={row['snn_bin_size']} status={row['status']}")
    if not dead:
        print("  none")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
