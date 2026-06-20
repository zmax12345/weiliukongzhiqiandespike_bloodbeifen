import argparse
import csv
import math
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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
from model import SNN_CNN_Hybrid


EPS = 1e-8

DEFAULT_CHECKPOINT_PATH = "/data/zm/2026.1.12_testdata/noblood/model/best_blood_flow_model.pth"
DEFAULT_MASK_PATH = "/data/zm/2026.1.12_testdata/noblood/blood_maskmosha_hot_pixel_mask.npy"
DEFAULT_OUTPUT_ROOT = "/data/zm/2026.1.12_testdata/noblood/loss_curve"

SOURCE_CONFIG = {
    "/data/zm/2026.1.12_testdata/1.15_150_680W": 0.010419,
    "/data/zm/2026.1.12_testdata/1.15_150_580W": 0.01139,
    "/data/zm/2026.1.12_testdata/2.3": 0.01001661,
    "/data/zm/2026.1.12_testdata/gaoyuzhi": 0.01449,
    "/data/zm/2026.1.12_testdata/1.26_PINN_result/2.4/data": 0.00987924,
    "/data/zm/2026.1.12_testdata/1.22data": 0.01,
}

SOURCE_SPLIT = {
    "/data/zm/2026.1.12_testdata/1.15_150_680W": "train",
    "/data/zm/2026.1.12_testdata/1.15_150_580W": "val",
    "/data/zm/2026.1.12_testdata/2.3": "val",
    "/data/zm/2026.1.12_testdata/gaoyuzhi": "train",
    "/data/zm/2026.1.12_testdata/1.26_PINN_result/2.4/data": "train",
    "/data/zm/2026.1.12_testdata/1.22data": "eval",
}

SOURCE_CONFIGS = [
    {"split": "train", "source_path": "/data/zm/2026.1.12_testdata/1.15_150_680W", "d": 0.010419},
    {"split": "val", "source_path": "/data/zm/2026.1.12_testdata/1.15_150_580W", "d": 0.01139},
    {"split": "val", "source_path": "/data/zm/2026.1.12_testdata/2.3", "d": 0.01001661},
    {"split": "train", "source_path": "/data/zm/2026.1.12_testdata/gaoyuzhi", "d": 0.01449},
    {"split": "train", "source_path": "/data/zm/2026.1.12_testdata/1.26_PINN_result/2.4/data", "d": 0.00987924},
    {"split": "eval", "source_path": "/data/zm/2026.1.12_testdata/1.22data", "d": 0.01},
]

PROXY_COLUMNS = [
    "raw_total_events",
    "normalized_total_events",
    "contrast2_0p8ms",
    "contrast2_1p6ms",
    "contrast2_3p2ms",
    "contrast2_6p4ms",
    "contrast2_12p8ms",
    "contrast_slope_logT",
    "contrast_area",
    "contrast_drop_ratio",
    "acf_lag40",
    "acf_lag80",
    "acf_decay_slope",
    "acf_integral",
    "low_freq_ratio_1ms",
    "low_freq_ratio_5ms",
    "spectral_centroid",
    "patch_total_cv",
    "patch_entropy",
    "patch_neighbor_corr",
    "patch_temporal_corr_mean",
    "spatial_blur_proxy",
]

TARGET_COLUMNS = [
    "velocity_true",
    "v_final",
    "residual",
    "abs_error",
    "tau_pred",
    "raw_total_events",
    "normalized_total_events",
]


def load_trusted_checkpoint(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def safe_float(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def source_name_from_path(source_path):
    return os.path.basename(os.path.normpath(str(source_path)))


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or y.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or y.size < 2:
        return float("nan")
    return safe_pearson(_rankdata(x), _rankdata(y))


def compute_scalar_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan")
    err = y_pred[finite] - y_true[finite]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true[finite]), EPS)) * 100.0)
    return mae, rmse, mape


def format_float(value, digits=6):
    value = safe_float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def finite_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std()) if values.size else float("nan")


def unpack_batch(batch):
    if len(batch) == 6:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = batch
    else:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids = batch
        metadata = [{} for _ in range(int(y_true.shape[0]))]
    return x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata


def aggregate_blocks_for_proxy(manager, batch_size, base_total_steps, base_block_size):
    patch_counts = []
    active_ratios = []
    num_blocks = int(math.ceil(base_total_steps / base_block_size))
    for block_idx in range(num_blocks):
        x_block = manager.get_block_dense(block_idx, base_block_size)
        if x_block.numel() == 0:
            continue
        steps = x_block.shape[0]
        patches_per_sample = manager.patches_per_sample
        patch_view = x_block.view(
            steps,
            batch_size,
            patches_per_sample,
            x_block.shape[2],
            x_block.shape[3],
            x_block.shape[4],
        )
        counts = patch_view.sum(dim=(3, 4, 5)).detach().cpu().numpy()
        patch_counts.append(counts)
        active = (patch_view > 0).float().mean(dim=(2, 3, 4, 5)).detach().cpu().numpy()
        active_ratios.append(active)
    if not patch_counts:
        return (
            np.zeros((0, batch_size, manager.patches_per_sample), dtype=np.float64),
            np.full((0, batch_size), np.nan, dtype=np.float64),
        )
    return np.concatenate(patch_counts, axis=0), np.concatenate(active_ratios, axis=0)


def acf_lag(counts, lag):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size <= lag:
        return float("nan")
    centered = counts - np.mean(counts)
    denom = np.var(counts) + EPS
    return float(np.mean(centered[:-lag] * centered[lag:]) / denom)


def linear_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2 or np.std(x[finite]) < 1e-12:
        return float("nan")
    return float(np.polyfit(x[finite], y[finite], 1)[0])


def patch_entropy(patch_total):
    total = float(np.sum(patch_total))
    if total <= 0:
        return float("nan")
    prob = patch_total / (total + EPS)
    return float(-np.sum(prob * np.log(prob + EPS)) / np.log(prob.size + EPS))


def adjacent_pair_indices(grid_rows, grid_cols):
    pairs = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            idx = row * grid_cols + col
            if col + 1 < grid_cols:
                pairs.append((idx, idx + 1))
            if row + 1 < grid_rows:
                pairs.append((idx, idx + grid_cols))
    return pairs


def neighbor_similarity_from_patch_total(patch_total, pairs):
    if not pairs:
        return float("nan")
    values = []
    mean_total = float(np.mean(patch_total))
    for a, b in pairs:
        denom = abs(patch_total[a]) + abs(patch_total[b]) + EPS
        values.append(1.0 - abs(patch_total[a] - patch_total[b]) / denom)
    return float(np.mean(values)) if values else float("nan")


def patch_temporal_corr(patch_series, pairs):
    if not pairs:
        return float("nan")
    values = []
    for a, b in pairs:
        corr = safe_pearson(patch_series[:, a], patch_series[:, b])
        if np.isfinite(corr):
            values.append(corr)
    return float(np.mean(values)) if values else float("nan")


def compute_sample_proxy(counts, patch_series, active_ratio, base_dt_us, grid_rows, grid_cols):
    result = {}
    counts = np.asarray(counts, dtype=np.float64)
    patch_series = np.asarray(patch_series, dtype=np.float64)
    active_ratio = np.asarray(active_ratio, dtype=np.float64)

    result["raw_total_events"] = float(np.sum(counts))
    result["mean_count_per_base_step"] = float(np.mean(counts)) if counts.size else float("nan")
    result["std_count_per_base_step"] = float(np.std(counts)) if counts.size else float("nan")
    result["cv_count_per_base_step"] = result["std_count_per_base_step"] / (result["mean_count_per_base_step"] + EPS)
    result["fano_base"] = float(np.var(counts) / (np.mean(counts) + EPS)) if counts.size else float("nan")
    result["nonzero_base_ratio"] = float(np.mean(counts > 0)) if counts.size else float("nan")
    result["active_pixel_ratio_mean"] = finite_mean(active_ratio)

    exposure_bins = [20, 40, 80, 160, 320, 640]
    contrast2_values = []
    exposure_ms_values = []
    for bins in exposure_bins:
        label = f"{bins * base_dt_us / 1000.0:g}ms".replace(".", "p")
        usable = (counts.size // bins) * bins
        if usable <= 0:
            exposure_count = np.asarray([], dtype=np.float64)
        else:
            exposure_count = counts[:usable].reshape(-1, bins).sum(axis=1)
        mean_exp = float(np.mean(exposure_count)) if exposure_count.size else float("nan")
        std_exp = float(np.std(exposure_count)) if exposure_count.size else float("nan")
        contrast = std_exp / (mean_exp + EPS) if np.isfinite(std_exp) else float("nan")
        contrast2 = contrast ** 2 if np.isfinite(contrast) else float("nan")
        result[f"contrast_{label}"] = contrast
        result[f"contrast2_{label}"] = contrast2
        result[f"fano_{label}"] = float(np.var(exposure_count) / (mean_exp + EPS)) if exposure_count.size else float("nan")
        result[f"nonzero_exp_ratio_{label}"] = float(np.mean(exposure_count > 0)) if exposure_count.size else float("nan")
        if np.isfinite(contrast2):
            exposure_ms_values.append(bins * base_dt_us / 1000.0)
            contrast2_values.append(contrast2)

    log_t = np.log(np.asarray(exposure_ms_values, dtype=np.float64) + EPS)
    log_c = np.log(np.asarray(contrast2_values, dtype=np.float64) + EPS)
    result["contrast_slope_logT"] = linear_slope(log_t, log_c)
    result["contrast_area"] = float(np.trapz(np.asarray(contrast2_values), log_t)) if len(contrast2_values) >= 2 else float("nan")
    c08 = result.get("contrast2_0p8ms", float("nan"))
    c128 = result.get("contrast2_12p8ms", float("nan"))
    result["contrast_drop_ratio"] = c128 / (c08 + EPS) if np.isfinite(c08) and np.isfinite(c128) else float("nan")
    if len(contrast2_values) >= 3:
        result["contrast_curvature"] = float(np.mean(np.diff(np.asarray(contrast2_values), n=2)))
    else:
        result["contrast_curvature"] = float("nan")

    acf_lags = [1, 5, 10, 20, 40, 80, 160]
    acf_values = []
    lag_ms_values = []
    for lag in acf_lags:
        value = acf_lag(counts, lag)
        result[f"acf_lag{lag}"] = value
        if np.isfinite(value):
            acf_values.append(value)
            lag_ms_values.append(lag * base_dt_us / 1000.0)
    result["acf_decay_slope"] = linear_slope(lag_ms_values, np.log(np.abs(acf_values) + EPS)) if acf_values else float("nan")
    result["acf_integral"] = float(np.sum(np.maximum(acf_values, 0.0))) if acf_values else float("nan")
    acf1 = result.get("acf_lag1", float("nan"))
    half_life = float("nan")
    if np.isfinite(acf1):
        for lag in acf_lags[1:]:
            value = result.get(f"acf_lag{lag}", float("nan"))
            if np.isfinite(value) and value < 0.5 * acf1:
                half_life = lag * base_dt_us / 1000.0
                break
    result["acf_half_life_proxy"] = half_life

    var_base = np.var(counts) + EPS
    for name, bins in (("1ms", 50), ("5ms", 250)):
        usable = (counts.size // bins) * bins
        if usable <= 0:
            result[f"low_freq_ratio_{name}"] = float("nan")
        else:
            smooth = counts[:usable].reshape(-1, bins).sum(axis=1)
            result[f"low_freq_ratio_{name}"] = float(np.var(smooth) / var_base)

    centered = counts - np.mean(counts) if counts.size else counts
    if centered.size >= 2 and np.any(np.isfinite(centered)):
        power = np.abs(np.fft.rfft(centered)) ** 2
        freqs = np.fft.rfftfreq(centered.size, d=base_dt_us * 1e-6)
        total_power = float(np.sum(power)) + EPS
        cutoff = np.nanmedian(freqs)
        result["high_freq_ratio"] = float(np.sum(power[freqs >= cutoff]) / total_power)
        result["low_freq_power_ratio"] = float(np.sum(power[freqs < cutoff]) / total_power)
        result["spectral_centroid"] = float(np.sum(freqs * power) / total_power)
    else:
        result["high_freq_ratio"] = float("nan")
        result["low_freq_power_ratio"] = float("nan")
        result["spectral_centroid"] = float("nan")

    patch_total = patch_series.sum(axis=0) if patch_series.size else np.asarray([], dtype=np.float64)
    result["patch_total_mean"] = float(np.mean(patch_total)) if patch_total.size else float("nan")
    result["patch_total_std"] = float(np.std(patch_total)) if patch_total.size else float("nan")
    result["patch_total_cv"] = result["patch_total_std"] / (result["patch_total_mean"] + EPS)
    result["patch_entropy"] = patch_entropy(patch_total) if patch_total.size else float("nan")
    pairs = adjacent_pair_indices(grid_rows, grid_cols)
    result["patch_neighbor_corr"] = neighbor_similarity_from_patch_total(patch_total, pairs) if patch_total.size else float("nan")
    result["patch_temporal_corr_mean"] = patch_temporal_corr(patch_series, pairs) if patch_series.size else float("nan")
    result["spatial_blur_proxy"] = result["patch_neighbor_corr"]
    return result


def compute_batch_proxies(manager, batch_size, base_total_steps, base_block_size, base_dt_us):
    patch_counts, active_ratios = aggregate_blocks_for_proxy(manager, batch_size, base_total_steps, base_block_size)
    rows = []
    for sample_idx in range(batch_size):
        patch_series = patch_counts[:, sample_idx, :] if patch_counts.size else np.zeros((0, manager.patches_per_sample))
        counts = patch_series.sum(axis=1)
        active = active_ratios[:, sample_idx] if active_ratios.size else np.asarray([], dtype=np.float64)
        rows.append(compute_sample_proxy(counts, patch_series, active, base_dt_us, manager.grid_rows, manager.grid_cols))
    return rows


def row_group_summary(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped[key].append(row)
    summaries = []
    for key, items in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        y_true = [item["velocity_true"] for item in items]
        y_pred = [item["v_final"] for item in items]
        mae, rmse, mape = compute_scalar_metrics(y_true, y_pred)
        summary = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        summary.update(
            {
                "samples": len(items),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "pred_mean": finite_mean(y_pred),
                "pred_std": finite_std(y_pred),
                "bias": finite_mean(np.asarray(y_pred) - np.asarray(y_true)),
            }
        )
        for col in (
            "raw_total_events",
            "contrast2_0p8ms",
            "contrast2_12p8ms",
            "contrast_slope_logT",
            "acf_integral",
            "low_freq_ratio_1ms",
            "spatial_blur_proxy",
            "patch_neighbor_corr",
        ):
            summary[f"{col}_mean"] = finite_mean([item.get(col, float("nan")) for item in items])
            summary[f"{col}_std"] = finite_std([item.get(col, float("nan")) for item in items])
        summaries.append(summary)
    return summaries


def correlation_table(rows, target, scope_name="all"):
    table = []
    y = np.asarray([row.get(target, float("nan")) for row in rows], dtype=np.float64)
    for proxy in PROXY_COLUMNS:
        x = np.asarray([row.get(proxy, float("nan")) for row in rows], dtype=np.float64)
        pearson = safe_pearson(x, y)
        spearman = safe_spearman(x, y)
        if np.isfinite(pearson) or np.isfinite(spearman):
            table.append(
                {
                    "scope": scope_name,
                    "target": target,
                    "proxy": proxy,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_spearman": abs(spearman) if np.isfinite(spearman) else float("nan"),
                }
            )
    return sorted(table, key=lambda r: safe_float(r["abs_spearman"], -1.0), reverse=True)


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "split",
        "source_name",
        "source_path",
        "file_path",
        "sample_index_within_source",
        "seq_start_idx",
        "velocity_true",
        "d_value",
        "v_final",
        "v_final_clipped",
        "residual",
        "abs_error",
        "tau_pred",
        "log_tau_pred",
        "v_pred_aux",
        "raw_total_events",
        "normalized_total_events",
        "source_scale",
    ]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(headers, rows, formatters=None, max_rows=None):
    formatters = formatters or {}
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows[:max_rows]:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if header in formatters:
                cells.append(formatters[header](value))
            elif isinstance(value, (float, np.floating)):
                cells.append(format_float(value))
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    if not rows:
        out.append("| " + " | ".join(["-"] * len(headers)) + " |")
    return "\n".join(out)


def build_scoped_correlations(rows):
    scoped = []
    scopes = [("all", rows)]
    for split in sorted({row.get("split", "unknown") for row in rows}):
        scopes.append((f"split={split}", [row for row in rows if row.get("split") == split]))
    for source in sorted({row.get("source_path", "") for row in rows}):
        scopes.append((f"source={os.path.basename(source)}", [row for row in rows if row.get("source_path") == source]))

    for scope_name, scope_rows in scopes:
        for target in TARGET_COLUMNS:
            scoped.extend(correlation_table(scope_rows, target, scope_name=scope_name))
    scoped.extend(correlation_table(rows, "is_val", scope_name="all"))
    scoped.extend(correlation_table(rows, "is_eval", scope_name="all"))
    scoped.extend(correlation_table(rows, "is_heldout", scope_name="all"))
    return scoped


def write_correlation_csv(path, rows):
    fields = ["scope", "target", "proxy", "pearson", "spearman", "abs_spearman"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_table_csv(path, rows, fields=None):
    if fields is None:
        fields = sorted({key for row in rows for key in row.keys()}) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _values(rows, column):
    return np.asarray([safe_float(row.get(column, np.nan)) for row in rows], dtype=np.float64)


def residualize_array(y, controls):
    y = np.asarray(y, dtype=np.float64)
    controls = [np.asarray(control, dtype=np.float64) for control in controls]
    finite = np.isfinite(y)
    for control in controls:
        finite &= np.isfinite(control)
    residual = np.full_like(y, np.nan, dtype=np.float64)
    if finite.sum() < len(controls) + 2:
        return residual
    x = np.column_stack([np.ones(int(finite.sum()), dtype=np.float64)] + [control[finite] for control in controls])
    beta, *_ = np.linalg.lstsq(x, y[finite], rcond=None)
    residual[finite] = y[finite] - x @ beta
    return residual


def event_count_control_values(rows):
    raw_total = _values(rows, "raw_total_events")
    finite_raw = raw_total[np.isfinite(raw_total)]
    if finite_raw.size >= 2 and finite_raw.std() > 1e-12:
        return raw_total
    return _values(rows, "normalized_total_events")


def compute_velocity_controlled_correlations(rows):
    detail = []
    velocities = sorted({safe_float(row.get("velocity_true")) for row in rows if np.isfinite(safe_float(row.get("velocity_true")))})
    for velocity in velocities:
        velocity_rows = [row for row in rows if np.isclose(safe_float(row.get("velocity_true")), velocity)]
        for proxy in PROXY_COLUMNS:
            x = _values(velocity_rows, proxy)
            for target in ("abs_error", "residual"):
                y = _values(velocity_rows, target)
                pearson = safe_pearson(x, y)
                spearman = safe_spearman(x, y)
                detail.append(
                    {
                        "velocity_true": velocity,
                        "proxy": proxy,
                        "target": target,
                        "samples": len(velocity_rows),
                        "pearson": pearson,
                        "spearman": spearman,
                        "abs_spearman": abs(spearman) if np.isfinite(spearman) else float("nan"),
                    }
                )

    summary = []
    for proxy in PROXY_COLUMNS:
        abs_rows = [row for row in detail if row["proxy"] == proxy and row["target"] == "abs_error"]
        residual_rows = [row for row in detail if row["proxy"] == proxy and row["target"] == "residual"]
        abs_vals = np.asarray([row["abs_spearman"] for row in abs_rows], dtype=np.float64)
        residual_vals = np.asarray([row["spearman"] for row in residual_rows], dtype=np.float64)
        abs_vals = abs_vals[np.isfinite(abs_vals)]
        residual_vals = residual_vals[np.isfinite(residual_vals)]
        summary.append(
            {
                "proxy": proxy,
                "mean_abs_spearman_with_abs_error": float(np.mean(abs_vals)) if abs_vals.size else float("nan"),
                "median_abs_spearman_with_abs_error": float(np.median(abs_vals)) if abs_vals.size else float("nan"),
                "max_abs_spearman_with_abs_error": float(np.max(abs_vals)) if abs_vals.size else float("nan"),
                "num_velocity_bins_abs_spearman_gt_0p4": int(np.sum(abs_vals > 0.4)) if abs_vals.size else 0,
                "mean_spearman_with_residual": float(np.mean(residual_vals)) if residual_vals.size else float("nan"),
            }
        )
    summary.sort(
        key=lambda row: (
            row["num_velocity_bins_abs_spearman_gt_0p4"],
            safe_float(row["mean_abs_spearman_with_abs_error"], -1.0),
        ),
        reverse=True,
    )
    return detail, summary


def compute_residualized_correlations(rows):
    velocity = _values(rows, "velocity_true")
    event_count = event_count_control_values(rows)
    abs_error_resid = residualize_array(_values(rows, "abs_error"), [velocity, event_count])
    residual_resid = residualize_array(_values(rows, "residual"), [velocity, event_count])
    table = []
    for proxy in PROXY_COLUMNS:
        proxy_resid = residualize_array(_values(rows, proxy), [velocity, event_count])
        abs_pearson = safe_pearson(proxy_resid, abs_error_resid)
        abs_spearman = safe_spearman(proxy_resid, abs_error_resid)
        residual_pearson = safe_pearson(proxy_resid, residual_resid)
        residual_spearman = safe_spearman(proxy_resid, residual_resid)
        table.append(
            {
                "proxy": proxy,
                "abs_error_resid_pearson": abs_pearson,
                "abs_error_resid_spearman": abs_spearman,
                "abs_error_resid_abs_spearman": abs(abs_spearman) if np.isfinite(abs_spearman) else float("nan"),
                "residual_resid_pearson": residual_pearson,
                "residual_resid_spearman": residual_spearman,
                "residual_resid_abs_spearman": abs(residual_spearman) if np.isfinite(residual_spearman) else float("nan"),
            }
        )
    table.sort(key=lambda row: safe_float(row["abs_error_resid_abs_spearman"], -1.0), reverse=True)
    return table


def summarize_source(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("source_name", ""), row.get("split", ""))].append(row)
    summaries = []
    for (source_name, split), items in sorted(grouped.items()):
        y_true = _values(items, "velocity_true")
        y_pred = _values(items, "v_final")
        mae, rmse, _ = compute_scalar_metrics(y_true, y_pred)
        summary = {
            "source_name": source_name,
            "split": split,
            "samples": len(items),
            "MAE": mae,
            "RMSE": rmse,
            "pred_mean": finite_mean(y_pred),
            "pred_std": finite_std(y_pred),
            "bias_mean": finite_mean(y_pred - y_true),
        }
        for col in (
            "raw_total_events",
            "normalized_total_events",
            "source_scale",
            "contrast_slope_logT",
            "contrast_area",
            "acf_integral",
            "spectral_centroid",
            "spatial_blur_proxy",
        ):
            summary[f"{col}_mean"] = finite_mean(_values(items, col))
            summary[f"{col}_std"] = finite_std(_values(items, col))
        summary["patch_temporal_corr_mean"] = finite_mean(_values(items, "patch_temporal_corr_mean"))
        summary["patch_temporal_corr_std"] = finite_std(_values(items, "patch_temporal_corr_mean"))
        summaries.append(summary)
    return summaries


def summarize_source_velocity(rows):
    grouped = defaultdict(list)
    for row in rows:
        velocity = safe_float(row.get("velocity_true"))
        grouped[(row.get("source_name", ""), row.get("split", ""), velocity)].append(row)
    summaries = []
    for (source_name, split, velocity), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        y_true = _values(items, "velocity_true")
        y_pred = _values(items, "v_final")
        mae, rmse, _ = compute_scalar_metrics(y_true, y_pred)
        summary = {
            "source_name": source_name,
            "split": split,
            "velocity_true": velocity,
            "samples": len(items),
            "MAE": mae,
            "RMSE": rmse,
            "pred_mean": finite_mean(y_pred),
            "bias": finite_mean(y_pred - y_true),
            "raw_total_events_mean": finite_mean(_values(items, "raw_total_events")),
            "raw_total_events_std": finite_std(_values(items, "raw_total_events")),
            "normalized_total_events_mean": finite_mean(_values(items, "normalized_total_events")),
            "normalized_total_events_std": finite_std(_values(items, "normalized_total_events")),
            "source_scale_mean": finite_mean(_values(items, "source_scale")),
            "contrast2_0p8ms_mean": finite_mean(_values(items, "contrast2_0p8ms")),
            "contrast2_12p8ms_mean": finite_mean(_values(items, "contrast2_12p8ms")),
            "contrast_slope_logT_mean": finite_mean(_values(items, "contrast_slope_logT")),
            "contrast_area_mean": finite_mean(_values(items, "contrast_area")),
            "acf_integral_mean": finite_mean(_values(items, "acf_integral")),
            "spectral_centroid_mean": finite_mean(_values(items, "spectral_centroid")),
            "patch_temporal_corr_mean": finite_mean(_values(items, "patch_temporal_corr_mean")),
            "spatial_blur_proxy_mean": finite_mean(_values(items, "spatial_blur_proxy")),
        }
        summaries.append(summary)
    return summaries


def compute_heldout_source_velocity_shift(rows):
    shifts = []
    velocities = sorted({safe_float(row.get("velocity_true")) for row in rows if np.isfinite(safe_float(row.get("velocity_true")))})
    for velocity in velocities:
        velocity_rows = [row for row in rows if np.isclose(safe_float(row.get("velocity_true")), velocity)]
        train_rows = [row for row in velocity_rows if row.get("split") == "train"]
        heldout_keys = sorted({
            (row.get("split", ""), row.get("source_name", ""))
            for row in velocity_rows
            if row.get("split") in {"val", "eval"}
        })
        for heldout_split, heldout_source in heldout_keys:
            heldout_rows = [
                row for row in velocity_rows
                if row.get("split") == heldout_split and row.get("source_name") == heldout_source
            ]
            heldout_true = _values(heldout_rows, "velocity_true")
            heldout_pred = _values(heldout_rows, "v_final")
            heldout_mae, _, _ = compute_scalar_metrics(heldout_true, heldout_pred)
            heldout_bias = finite_mean(heldout_pred - heldout_true)
            for proxy in PROXY_COLUMNS:
                train_vals = _values(train_rows, proxy)
                heldout_vals = _values(heldout_rows, proxy)
                train_mean = finite_mean(train_vals)
                train_std = finite_std(train_vals)
                heldout_mean = finite_mean(heldout_vals)
                train_denom = train_std if np.isfinite(train_std) and train_std > EPS else EPS
                shifts.append(
                    {
                        "heldout_split": heldout_split,
                        "heldout_source": heldout_source,
                        "velocity_true": velocity,
                        "proxy": proxy,
                        "train_mean": train_mean,
                        "train_std": train_std,
                        "heldout_mean": heldout_mean,
                        "heldout_minus_train": heldout_mean - train_mean if np.isfinite(heldout_mean) and np.isfinite(train_mean) else float("nan"),
                        "heldout_z_vs_train": (heldout_mean - train_mean) / train_denom if np.isfinite(heldout_mean) and np.isfinite(train_mean) else float("nan"),
                        "heldout_mae": heldout_mae,
                        "heldout_bias": heldout_bias,
                        "train_samples": len(train_rows),
                        "heldout_samples": len(heldout_rows),
                    }
                )
    shifts.sort(key=lambda row: abs(safe_float(row.get("heldout_z_vs_train"), 0.0)), reverse=True)
    return shifts


def compute_top_problem_bins(source_velocity_summary, splits):
    wanted = set(splits)
    rows = [row for row in source_velocity_summary if row.get("split") in wanted]
    return sorted(rows, key=lambda row: safe_float(row.get("MAE"), -1.0), reverse=True)


def source_indicator_correlations(rows):
    result = []
    for indicator in ("is_val", "is_eval", "is_heldout"):
        result.extend(correlation_table(rows, indicator, scope_name="all"))
    return sorted(result, key=lambda row: safe_float(row.get("abs_spearman"), -1.0), reverse=True)


def residualized_series_for_proxy(rows, proxy):
    velocity = _values(rows, "velocity_true")
    event_count = event_count_control_values(rows)
    return (
        residualize_array(_values(rows, proxy), [velocity, event_count]),
        residualize_array(_values(rows, "abs_error"), [velocity, event_count]),
    )


def write_report(path, config, rows, correlations, scoped_correlations, per_source, per_velocity, per_source_velocity, plot_paths, elapsed):
    abs_corr = correlations.get("abs_error", [])
    residual_corr = correlations.get("residual", [])
    eval_indicator_corr = correlations.get("is_eval_source", [])
    lines = [
        "# Multiple Scattering Proxy Diagnostic",
        "",
        "This script is diagnostic only.",
        "It does not train or modify the model.",
        "It searches for multiple-scattering proxies from event statistics.",
        "",
        "## Run Config",
        "",
        f"- timestamp: `{config['timestamp']}`",
        f"- elapsed: `{elapsed:.2f}s`",
        f"- device: `{config['device']}`",
        f"- checkpoint_path: `{config['checkpoint_path']}`",
        f"- checkpoint_stage: `{config['checkpoint_stage']}`",
        f"- event_norm_mode: `{config['event_norm_mode']}`",
        f"- event_norm_stats_from_checkpoint: `{config['event_norm_stats_from_checkpoint']}`",
        f"- base_dt_us: `{config['base_dt_us']}`",
        f"- window_ms: `{config['window_ms']}`",
        f"- base_total_steps: `{config['base_total_steps']}`",
        f"- base_block_size: `{config['base_block_size']}`",
        f"- snn_bin_size: `{config['snn_bin_size']}`",
        f"- snn_step_us: `{config['snn_step_us']}`",
        f"- snn_steps: `{config['snn_steps']}`",
        f"- snn_input_scale_mode: `{config['snn_input_scale_mode']}`",
        f"- samples: `{len(rows)}`",
        "",
        "## Overall Metrics",
        "",
    ]
    mae, rmse, mape = compute_scalar_metrics([r["velocity_true"] for r in rows], [r["v_final"] for r in rows])
    lines.extend(
        [
            f"- MAE: `{format_float(mae)}`",
            f"- RMSE: `{format_float(rmse)}`",
            f"- MAPE: `{format_float(mape, 2)}%`",
            "",
            "## Per-Source Summary",
            "",
            markdown_table(
                [
                    "source_path",
                    "samples",
                    "mae",
                    "pred_mean",
                    "pred_std",
                    "raw_total_events_mean",
                    "raw_total_events_std",
                    "contrast2_0p8ms_mean",
                    "contrast2_12p8ms_mean",
                    "contrast_slope_logT_mean",
                    "acf_integral_mean",
                    "low_freq_ratio_1ms_mean",
                    "spatial_blur_proxy_mean",
                    "patch_neighbor_corr_mean",
                ],
                per_source,
            ),
            "",
            "## Per-Velocity Summary",
            "",
            markdown_table(
                [
                    "velocity_true",
                    "samples",
                    "mae",
                    "pred_mean",
                    "bias",
                    "contrast2_0p8ms_mean",
                    "contrast2_12p8ms_mean",
                    "contrast_slope_logT_mean",
                    "acf_integral_mean",
                    "low_freq_ratio_1ms_mean",
                    "spatial_blur_proxy_mean",
                ],
                per_velocity,
            ),
            "",
            "## Per-Source-Per-Velocity Summary",
            "",
            markdown_table(
                [
                    "source_path",
                    "velocity_true",
                    "samples",
                    "mae",
                    "bias",
                    "contrast_slope_logT_mean",
                    "acf_integral_mean",
                    "spatial_blur_proxy_mean",
                ],
                per_source_velocity,
            ),
            "",
            "## Top Proxies Correlated With Absolute Error",
            "",
            markdown_table(["proxy", "pearson", "spearman"], abs_corr, max_rows=20),
            "",
            "## Top Proxies Correlated With Residual",
            "",
            markdown_table(["proxy", "pearson", "spearman"], residual_corr, max_rows=20),
            "",
            "## Top Proxies Correlated With Eval Source Indicator",
            "",
            markdown_table(["proxy", "pearson", "spearman"], eval_indicator_corr, max_rows=20),
            "",
            "## Scoped Correlation Highlights",
            "",
            markdown_table(
                ["scope", "target", "proxy", "pearson", "spearman"],
                sorted(scoped_correlations, key=lambda r: safe_float(r.get("abs_spearman"), -1.0), reverse=True),
                max_rows=40,
            ),
            "",
            "## Plots",
            "",
        ]
    )
    for name, plot_path in plot_paths.items():
        lines.append(f"- {name}: `{plot_path}`")

    checklist = []
    source_shift = False
    train_sources = [r for r in per_source if SOURCE_SPLIT.get(r.get("source_path"), "") == "train"]
    eval_sources = [r for r in per_source if SOURCE_SPLIT.get(r.get("source_path"), "") == "eval"]
    if train_sources and eval_sources:
        train_blur = finite_mean([r.get("spatial_blur_proxy_mean") for r in train_sources])
        eval_blur = finite_mean([r.get("spatial_blur_proxy_mean") for r in eval_sources])
        train_contrast = finite_mean([r.get("contrast_slope_logT_mean") for r in train_sources])
        eval_contrast = finite_mean([r.get("contrast_slope_logT_mean") for r in eval_sources])
        if abs(eval_blur - train_blur) > 0.05 or abs(eval_contrast - train_contrast) > 0.2:
            source_shift = True
    if source_shift:
        checklist.append("Potential source-level scattering shift detected.")

    for table, target_name in ((abs_corr, "absolute error"), (residual_corr, "residual")):
        for row in table:
            if np.isfinite(row["spearman"]) and abs(row["spearman"]) > 0.4:
                checklist.append(f"Proxy `{row['proxy']}` is moderately correlated with {target_name}.")
                break

    spatial_hits = [
        row for row in abs_corr + eval_indicator_corr
        if row["proxy"] in {"spatial_blur_proxy", "patch_neighbor_corr", "patch_temporal_corr_mean"}
        and np.isfinite(row["spearman"]) and abs(row["spearman"]) > 0.4
    ]
    if spatial_hits:
        checklist.append("Spatial mixing / multiple scattering proxy may explain residual errors.")

    temporal_hits = [
        row for row in abs_corr + eval_indicator_corr
        if row["proxy"] in {"acf_integral", "acf_lag40", "acf_lag80", "acf_decay_slope"}
        and np.isfinite(row["spearman"]) and abs(row["spearman"]) > 0.4
    ]
    if temporal_hits:
        checklist.append("Long-tail temporal autocorrelation may indicate slow/static or multiple-scattering contribution.")

    contrast_hits = [
        row for row in abs_corr + residual_corr + eval_indicator_corr
        if row["proxy"].startswith("contrast") or row["proxy"].startswith("acf")
    ]
    if any(np.isfinite(row["spearman"]) and abs(row["spearman"]) > 0.4 for row in contrast_hits):
        checklist.append("Multi-scale event contrast consistency is a promising physics-informed constraint.")

    if not checklist:
        checklist.append(
            "No strong multiple-scattering proxy was detected from these simple statistics; "
            "more explicit optical modeling or patch-level contrast may be needed."
        )

    lines.extend(["", "## Interpretation Checklist", ""])
    for item in checklist:
        lines.append(f"- {item}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_plots(rows, output_dir):
    plot_paths = {}
    if plt is None or not rows:
        return plot_paths
    sources = sorted({row["source_path"] for row in rows})
    velocities = sorted({row["velocity_true"] for row in rows})

    exposure_cols = ["contrast2_0p8ms", "contrast2_1p6ms", "contrast2_3p2ms", "contrast2_6p4ms", "contrast2_12p8ms"]
    fig, axes = plt.subplots(len(sources), 1, figsize=(10, max(4, 3 * len(sources))), squeeze=False)
    for ax, source in zip(axes[:, 0], sources):
        source_rows = [row for row in rows if row["source_path"] == source]
        for col in exposure_cols:
            means = []
            for velocity in velocities:
                means.append(finite_mean([r.get(col) for r in source_rows if r["velocity_true"] == velocity]))
            ax.plot(velocities, means, marker="o", label=col)
        ax.set_title(source)
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Mean contrast2")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "proxy_velocity_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    plot_paths["proxy_velocity_curves"] = path

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    scatter_specs = [
        ("contrast_slope_logT", "abs_error"),
        ("spatial_blur_proxy", "abs_error"),
        ("acf_integral", "abs_error"),
        ("low_freq_ratio_1ms", "abs_error"),
    ]
    for ax, (x_col, y_col) in zip(axes.ravel(), scatter_specs):
        ax.scatter([r.get(x_col, np.nan) for r in rows], [r.get(y_col, np.nan) for r in rows], s=12, alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "proxy_residual_correlations.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    plot_paths["proxy_residual_correlations"] = path

    box_cols = ["contrast_slope_logT", "acf_integral", "spatial_blur_proxy", "low_freq_ratio_1ms"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.ravel(), box_cols):
        data = [
            [r.get(col, np.nan) for r in rows if r["source_path"] == source and np.isfinite(r.get(col, np.nan))]
            for source in sources
        ]
        ax.boxplot(data, labels=[os.path.basename(s) for s in sources], showfliers=False)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "proxy_source_boxplots.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    plot_paths["proxy_source_boxplots"] = path

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for source in sources:
        source_rows = [row for row in rows if row["source_path"] == source]
        maes = []
        biases = []
        for velocity in velocities:
            part = [r for r in source_rows if r["velocity_true"] == velocity]
            if part:
                y_true = [r["velocity_true"] for r in part]
                y_pred = [r["v_final"] for r in part]
                mae, _, _ = compute_scalar_metrics(y_true, y_pred)
                maes.append(mae)
                biases.append(finite_mean(np.asarray(y_pred) - np.asarray(y_true)))
            else:
                maes.append(np.nan)
                biases.append(np.nan)
        axes[0].plot(velocities, maes, marker="o", label=os.path.basename(source))
        axes[1].plot(velocities, biases, marker="o", label=os.path.basename(source))
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Bias")
    axes[1].set_xlabel("Velocity")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "prediction_error_by_velocity.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    plot_paths["prediction_error_by_velocity"] = path
    return plot_paths


def make_plots_v2(rows, output_dir, residualized_correlations, heldout_source_velocity_shift):
    plot_paths = {}
    if plt is None or not rows:
        return plot_paths

    sources = sorted({row.get("source_name", "") for row in rows if row.get("source_name", "")})
    velocities = sorted({safe_float(row.get("velocity_true")) for row in rows if np.isfinite(safe_float(row.get("velocity_true")))})

    try:
        fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        curve_proxies = ["contrast_slope_logT", "acf_integral", "spectral_centroid"]
        for ax, proxy in zip(axes, curve_proxies):
            for source in sources:
                means = []
                for velocity in velocities:
                    vals = [
                        row.get(proxy, np.nan)
                        for row in rows
                        if row.get("source_name") == source and np.isclose(safe_float(row.get("velocity_true")), velocity)
                    ]
                    means.append(finite_mean(vals))
                ax.plot(velocities, means, marker="o", label=source)
            ax.set_ylabel(proxy)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Velocity")
        axes[0].legend()
        fig.tight_layout()
        path = os.path.join(output_dir, "proxy_velocity_curves_by_source.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths["proxy_velocity_curves_by_source"] = path
    except Exception as exc:
        print(f"Warning: failed to create proxy velocity curves: {exc}")

    try:
        selected = [
            "raw_total_events",
            "normalized_total_events",
            "contrast_slope_logT",
            "contrast_area",
            "acf_integral",
            "spectral_centroid",
            "patch_temporal_corr_mean",
            "spatial_blur_proxy",
            "patch_total_cv",
            "patch_entropy",
        ]
        row_keys = sorted({
            (row.get("heldout_split", ""), row.get("heldout_source", ""), safe_float(row.get("velocity_true")))
            for row in heldout_source_velocity_shift
            if np.isfinite(safe_float(row.get("velocity_true")))
        })
        matrix = np.full((len(row_keys), len(selected)), np.nan, dtype=np.float64)
        for i, (heldout_split, heldout_source, velocity) in enumerate(row_keys):
            for j, proxy in enumerate(selected):
                match = [
                    row
                    for row in heldout_source_velocity_shift
                    if row.get("proxy") == proxy and np.isclose(safe_float(row.get("velocity_true")), velocity)
                    and row.get("heldout_split") == heldout_split
                    and row.get("heldout_source") == heldout_source
                ]
                if match:
                    matrix[i, j] = safe_float(match[0].get("heldout_z_vs_train"))
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(np.arange(len(selected)))
        ax.set_xticklabels(selected, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_keys)))
        ax.set_yticklabels([f"{split}/{source}/{format_float(velocity, 3)}" for split, source, velocity in row_keys])
        ax.set_xlabel("Proxy")
        ax.set_ylabel("Heldout split/source/velocity")
        ax.set_title("Heldout z-shift vs train within matched velocity")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        path = os.path.join(output_dir, "heldout_shift_heatmap.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths["heldout_shift_heatmap"] = path
    except Exception as exc:
        print(f"Warning: failed to create heldout shift heatmap: {exc}")

    try:
        top = [row["proxy"] for row in residualized_correlations[:3] if np.isfinite(safe_float(row.get("abs_error_resid_abs_spearman")))]
        if top:
            fig, axes = plt.subplots(1, len(top), figsize=(5 * len(top), 4), squeeze=False)
            for ax, proxy in zip(axes[0], top):
                x_resid, y_resid = residualized_series_for_proxy(rows, proxy)
                ax.scatter(x_resid, y_resid, s=12, alpha=0.7)
                ax.set_xlabel(f"{proxy} residual")
                ax.set_ylabel("abs_error residual")
                ax.grid(True, alpha=0.25)
            fig.tight_layout()
            path = os.path.join(output_dir, "residualized_proxy_scatter.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plot_paths["residualized_proxy_scatter"] = path
    except Exception as exc:
        print(f"Warning: failed to create residualized scatter: {exc}")

    try:
        proxy = residualized_correlations[0]["proxy"] if residualized_correlations else "contrast_slope_logT"
        eval_rows = [row for row in rows if row.get("split") == "eval"]
        x = _values(eval_rows, proxy)
        y = _values(eval_rows, "abs_error")
        velocity = _values(eval_rows, "velocity_true")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(x, y, c=velocity, s=16, alpha=0.75)
        ax.set_xlabel(proxy)
        ax.set_ylabel("abs_error")
        ax.set_title("Eval error vs proxy")
        fig.colorbar(sc, ax=ax, label="velocity")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = os.path.join(output_dir, "eval_error_proxy_scatter.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths["eval_error_proxy_scatter"] = path
    except Exception as exc:
        print(f"Warning: failed to create eval error scatter: {exc}")

    return plot_paths


def _top_source_shift_proxy_counts(source_velocity_shift, threshold=1.0):
    counts = defaultdict(int)
    for row in source_velocity_shift:
        z = safe_float(row.get("heldout_z_vs_train"))
        if np.isfinite(z) and abs(z) > threshold:
            counts[row.get("proxy", "")] += 1
    return counts


PROBLEM_BIN_HEADERS = [
    "split",
    "source_name",
    "velocity_true",
    "samples",
    "MAE",
    "RMSE",
    "bias",
    "pred_mean",
    "raw_total_events_mean",
    "normalized_total_events_mean",
    "contrast_slope_logT_mean",
    "contrast_area_mean",
    "acf_integral_mean",
    "spectral_centroid_mean",
    "patch_temporal_corr_mean",
    "spatial_blur_proxy_mean",
]


def append_problematic_bins_section(lines, title, rows, empty_text):
    lines.extend([title, ""])
    if rows:
        lines.append(markdown_table(PROBLEM_BIN_HEADERS, rows, max_rows=30))
    else:
        lines.append(empty_text)
    lines.append("")


def write_report_v2(
    path,
    config,
    rows,
    correlations,
    source_indicator_corr,
    velocity_controlled_summary,
    residualized_correlations,
    per_source,
    source_velocity_summary,
    top_val_bins,
    top_eval_bins,
    top_heldout_bins,
    heldout_source_velocity_shift,
    plot_paths,
    elapsed,
):
    abs_corr = correlations.get("abs_error", [])
    residual_corr = correlations.get("residual", [])
    source_shift_counts = _top_source_shift_proxy_counts(heldout_source_velocity_shift)
    residualized_candidates = {
        row["proxy"]
        for row in residualized_correlations
        if safe_float(row.get("abs_error_resid_abs_spearman"), 0.0) > 0.3
    }
    source_shift_candidates = {proxy for proxy, count in source_shift_counts.items() if count >= 3}
    strong_candidates = sorted(residualized_candidates & source_shift_candidates)

    spatial_candidates = {"patch_neighbor_corr", "spatial_blur_proxy"}
    spatial_evidence = bool((residualized_candidates | source_shift_candidates) & spatial_candidates)

    lines = [
        "# Multiple Scattering Proxy Diagnostic",
        "",
        "This script is diagnostic only.",
        "It does not train or modify the model.",
        "It searches for multiple-scattering proxies from event statistics.",
        "",
        "## Run Config",
        "",
        f"- Timestamp: {config['timestamp']}",
        f"- Elapsed seconds: {elapsed:.1f}",
        f"- Device: {config['device']}",
        f"- Checkpoint: {config['checkpoint_path']}",
        f"- Checkpoint stage: {config.get('checkpoint_stage', 'unknown')}",
        f"- Event norm mode: {config['event_norm_mode']}",
        f"- Event norm clip: {config['event_norm_clip']}",
        f"- Event norm stats from checkpoint: {config.get('event_norm_stats_from_checkpoint', False)}",
        f"- Base dt us: {config['base_dt_us']}",
        f"- Window ms: {config['window_ms']}",
        f"- Base total steps: {config['base_total_steps']}",
        f"- Base block size: {config['base_block_size']}",
        f"- SNN bin size: {config['snn_bin_size']}",
        f"- SNN step us: {config['snn_step_us']}",
        f"- SNN steps: {config['snn_steps']}",
        f"- SNN input scale mode: {config['snn_input_scale_mode']}",
        f"- Batch size: {config['batch_size']}",
        f"- Max batches per source: {config['max_batches']}",
        "",
        "## Event Count Columns",
        "",
        "- raw_total_events comes from dataset metadata before event normalization.",
        "- normalized_total_events is computed from the actual model input after source_scale/event normalization.",
        "- raw_total_events unavailable from current loader." if not np.isfinite(_values(rows, "raw_total_events")).any() else "- raw_total_events metadata is available.",
        "",
        "## Sources",
        "",
        markdown_table(["split", "source_name", "source_path", "d"], [
            {
                "split": item["split"],
                "source_name": source_name_from_path(item["source_path"]),
                "source_path": item["source_path"],
                "d": item["d"],
            }
            for item in SOURCE_CONFIGS
        ]),
        "",
        "## Per-source Summary",
        "",
        markdown_table(
            [
                "source_name",
                "split",
                "samples",
                "MAE",
                "RMSE",
                "pred_mean",
                "pred_std",
                "bias_mean",
                "raw_total_events_mean",
                "raw_total_events_std",
                "normalized_total_events_mean",
                "normalized_total_events_std",
                "source_scale_mean",
                "source_scale_std",
                "contrast_slope_logT_mean",
                "contrast_slope_logT_std",
                "contrast_area_mean",
                "contrast_area_std",
                "acf_integral_mean",
                "acf_integral_std",
                "spectral_centroid_mean",
                "spectral_centroid_std",
                "patch_temporal_corr_mean",
                "patch_temporal_corr_std",
                "spatial_blur_proxy_mean",
                "spatial_blur_proxy_std",
            ],
            per_source,
        ),
        "",
        "## Per-source-per-velocity Summary",
        "",
        markdown_table(
            [
                "source_name",
                "split",
                "velocity_true",
                "samples",
                "MAE",
                "RMSE",
                "pred_mean",
                "bias",
                "raw_total_events_mean",
                "normalized_total_events_mean",
                "source_scale_mean",
                "contrast2_0p8ms_mean",
                "contrast2_12p8ms_mean",
                "contrast_slope_logT_mean",
                "contrast_area_mean",
                "acf_integral_mean",
                "spectral_centroid_mean",
                "patch_temporal_corr_mean",
                "spatial_blur_proxy_mean",
            ],
            source_velocity_summary,
            max_rows=80,
        ),
        "",
        "## Top Proxies Correlated With Absolute Error",
        "",
        markdown_table(["proxy", "pearson", "spearman", "abs_spearman"], abs_corr, max_rows=20),
        "",
        "## Top Proxies Correlated With Residual",
        "",
        markdown_table(["proxy", "pearson", "spearman", "abs_spearman"], residual_corr, max_rows=20),
        "",
        "## Source / Heldout Indicator Correlations",
        "",
        markdown_table(["target", "proxy", "pearson", "spearman", "abs_spearman"], source_indicator_corr, max_rows=30),
        "",
        "## Velocity-Controlled Proxy Correlations",
        "",
        markdown_table(
            [
                "proxy",
                "mean_abs_spearman_with_abs_error",
                "median_abs_spearman_with_abs_error",
                "max_abs_spearman_with_abs_error",
                "num_velocity_bins_abs_spearman_gt_0p4",
                "mean_spearman_with_residual",
            ],
            velocity_controlled_summary,
            max_rows=30,
        ),
        "",
        "## Residualized Correlations Controlling Velocity and Event Count",
        "",
        markdown_table(
            [
                "proxy",
                "abs_error_resid_pearson",
                "abs_error_resid_spearman",
                "abs_error_resid_abs_spearman",
                "residual_resid_pearson",
                "residual_resid_spearman",
            ],
            residualized_correlations,
            max_rows=30,
        ),
        "",
        "## Heldout Source Shift Within Same Velocity",
        "",
        markdown_table(
            [
                "heldout_split",
                "heldout_source",
                "velocity_true",
                "proxy",
                "train_mean",
                "train_std",
                "heldout_mean",
                "heldout_minus_train",
                "heldout_z_vs_train",
                "heldout_mae",
                "heldout_bias",
            ],
            heldout_source_velocity_shift,
            max_rows=60,
        ),
        "",
        "## Plots",
        "",
    ]
    problem_lines = []
    append_problematic_bins_section(problem_lines, "## Top Val Problematic Bins", top_val_bins, "No val samples found.")
    append_problematic_bins_section(problem_lines, "## Top Eval Problematic Bins", top_eval_bins, "No eval samples found.")
    append_problematic_bins_section(problem_lines, "## Top Heldout Problematic Bins", top_heldout_bins, "No heldout samples found.")
    try:
        insert_at = lines.index("## Top Proxies Correlated With Absolute Error")
        lines[insert_at:insert_at] = problem_lines
    except ValueError:
        lines.extend(problem_lines)

    if plot_paths:
        for name, plot_path in plot_paths.items():
            lines.append(f"- {name}: `{plot_path}`")
    else:
        lines.append("- Plot generation skipped or unavailable.")

    lines.extend(["", "## Interpretation Checklist", ""])
    pre_control = [row for row in abs_corr if safe_float(row.get("abs_spearman"), 0.0) > 0.4]
    if pre_control:
        top = pre_control[0]
        lines.append(
            f"- Event-dynamics proxy {top['proxy']} is correlated with absolute error before controlling velocity "
            f"(Spearman={format_float(top.get('spearman'))})."
        )

    vc = [
        row
        for row in velocity_controlled_summary
        if int(row.get("num_velocity_bins_abs_spearman_gt_0p4", 0)) >= 3
    ]
    for row in vc[:5]:
        lines.append(f"- Proxy {row['proxy']} remains related to error within multiple velocity bins.")

    for row in residualized_correlations[:5]:
        if safe_float(row.get("abs_error_resid_abs_spearman"), 0.0) > 0.3:
            lines.append(
                f"- Proxy {row['proxy']} remains correlated with error after controlling velocity and event count "
                f"(Spearman={format_float(row.get('abs_error_resid_spearman'))})."
            )

    for proxy, count in sorted(source_shift_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        if count >= 3:
            lines.append(f"- Proxy {proxy} shows consistent heldout-source shift within matched velocities.")

    for proxy in strong_candidates:
        lines.append(f"- Strong candidate multiple-scattering proxy detected: {proxy}.")

    if not strong_candidates:
        lines.append(
            "- No independently validated multiple-scattering proxy was confirmed. "
            "Current proxies may still be velocity/intensity/source-dynamics proxies."
        )

    if spatial_evidence:
        lines.append("- Spatial blur / spatial mixing evidence detected.")

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_config_from_checkpoint(checkpoint, args, device):
    config = {
        "base_dt_us": 20,
        "window_ms": 200,
        "base_total_steps": 10000,
        "base_block_size": 400,
        "snn_bin_size": 40,
        "snn_step_us": 800,
        "snn_steps": 250,
        "snn_input_scale_mode": "sqrt",
        "event_norm_mode": "source_scale",
        "event_norm_clip": (0.25, 4.0),
        "max_velocity": 2.0,
        "spatial_shape": (100, 368),
        "patch_shape": (50, 46),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "checkpoint_path": args.checkpoint_path,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "checkpoint_stage": "unknown",
        "event_norm_stats_from_checkpoint": False,
    }
    if isinstance(checkpoint, dict):
        input_config = checkpoint.get("input_config", {})
        for key in (
            "base_dt_us",
            "window_ms",
            "base_total_steps",
            "base_block_size",
            "snn_bin_size",
            "snn_step_us",
            "snn_steps",
            "snn_input_scale_mode",
        ):
            if key in input_config:
                config[key] = input_config[key]
        config["checkpoint_stage"] = checkpoint.get("stage", "unknown")
        event_norm_config = checkpoint.get("event_norm_config", {})
        config["max_velocity"] = event_norm_config.get("max_velocity", checkpoint.get("max_velocity", config["max_velocity"]))
        if "event_norm_mode" in event_norm_config:
            config["event_norm_mode"] = event_norm_config["event_norm_mode"]
        if "event_norm_clip" in event_norm_config:
            config["event_norm_clip"] = tuple(event_norm_config["event_norm_clip"])
        config["event_norm_stats_from_checkpoint"] = checkpoint.get("event_norm_stats") is not None
    return config


def main():
    parser = argparse.ArgumentParser(description="Analyze multiple-scattering proxy statistics without training.")
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--mask-path", default=DEFAULT_MASK_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_trusted_checkpoint(args.checkpoint_path, map_location=device)
    config = build_config_from_checkpoint(checkpoint, args, device)
    timestamp = config["timestamp"]
    output_dir = os.path.join(args.output_root, f"multiscattering_proxy_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_event_norm_stats = checkpoint.get("event_norm_stats") if isinstance(checkpoint, dict) else None
    checkpoint_reference_mean = (
        checkpoint_event_norm_stats.get("reference_mean_events_per_sample")
        if checkpoint_event_norm_stats is not None
        else None
    )

    model = SNN_CNN_Hybrid(in_channels=1, max_velocity=float(config["max_velocity"])).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    rows = []
    with torch.no_grad():
        for source_cfg in SOURCE_CONFIGS:
            source_path = source_cfg["source_path"]
            source_name = source_name_from_path(source_path)
            split = source_cfg["split"]
            source_dataset = FlexibleBloodFlowDataset(
                {source_path: source_cfg["d"]},
                mask_path=args.mask_path,
                T=1,
                seq_len=int(config["base_total_steps"]),
                dt_us=int(config["base_dt_us"]),
                max_velocity=float(config["max_velocity"]),
                event_norm_mode=config["event_norm_mode"],
                event_norm_stats=checkpoint_event_norm_stats,
                event_norm_reference_mean=checkpoint_reference_mean,
                event_norm_clip=config["event_norm_clip"],
                event_intensity_jitter_range=None,
                return_metadata=True,
            )
            loader = DataLoader(
                source_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=sequence_sparse_collate,
                num_workers=0,
            )
            total_batches = len(loader) if args.max_batches is None else min(args.max_batches, len(loader))
            desc = f"Analyze {source_name} ({split})"
            progress = tqdm(enumerate(loader), total=total_batches, desc=desc, dynamic_ncols=True)
            sample_offset = 0
            for batch_idx, batch in progress:
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = unpack_batch(batch)
                d_values_device = d_values.to(device)
                manager = DenseBlockManager(
                    x_seq_sparse_data,
                    batch_size=int(y_true.shape[0]),
                    spatial_shape=tuple(config["spatial_shape"]),
                    patch_shape=tuple(config["patch_shape"]),
                )
                output = model(
                    dataloader_or_generator=manager,
                    base_total_steps=int(config["base_total_steps"]),
                    base_block_size=int(config["base_block_size"]),
                    snn_bin_size=int(config["snn_bin_size"]),
                    snn_input_scale_mode=config["snn_input_scale_mode"],
                    base_dt_us=int(config["base_dt_us"]),
                )
                tau_pred = output["tau_pred"]
                log_tau_pred = output["log_tau_pred"]
                v_pred_aux = output.get("v_pred", torch.full_like(tau_pred, float("nan")))
                v_final = d_values_device / torch.clamp(tau_pred, min=1e-8)
                v_final_clipped = torch.clamp(v_final, min=0.0, max=float(config["max_velocity"]))
                proxy_rows = compute_batch_proxies(
                    manager,
                    batch_size=int(y_true.shape[0]),
                    base_total_steps=int(config["base_total_steps"]),
                    base_block_size=int(config["base_block_size"]),
                    base_dt_us=int(config["base_dt_us"]),
                )

                y_np = y_true.detach().cpu().numpy()
                d_np = d_values.detach().cpu().numpy()
                v_np = v_final.detach().cpu().numpy()
                v_clip_np = v_final_clipped.detach().cpu().numpy()
                tau_np = tau_pred.detach().cpu().numpy()
                log_tau_np = log_tau_pred.detach().cpu().numpy()
                aux_np = v_pred_aux.detach().cpu().numpy()

                for sample_idx in range(int(y_true.shape[0])):
                    meta = metadata[sample_idx] if sample_idx < len(metadata) and isinstance(metadata[sample_idx], dict) else {}
                    proxy_row = dict(proxy_rows[sample_idx])
                    normalized_total_events = safe_float(proxy_row.get("raw_total_events", np.nan))
                    proxy_row["normalized_total_events"] = normalized_total_events
                    metadata_raw_total = safe_float(meta.get("raw_total_events", np.nan))
                    raw_total_events = metadata_raw_total if np.isfinite(metadata_raw_total) else float("nan")
                    source_scale = safe_float(meta.get("source_scale", np.nan))
                    row = {
                        "split": split,
                        "source_name": source_name,
                        "source_path": source_path,
                        "file_path": meta.get("file_path", ""),
                        "sample_index_within_source": sample_offset + sample_idx,
                        "seq_start_idx": meta.get("seq_start_idx", ""),
                        "velocity_true": float(y_np[sample_idx]),
                        "d_value": float(source_cfg["d"]),
                        "batch_d_value": float(d_np[sample_idx]),
                        "v_final": float(v_np[sample_idx]),
                        "v_final_clipped": float(v_clip_np[sample_idx]),
                        "residual": float(v_np[sample_idx] - y_np[sample_idx]),
                        "abs_error": float(abs(v_np[sample_idx] - y_np[sample_idx])),
                        "tau_pred": float(tau_np[sample_idx]),
                        "log_tau_pred": float(log_tau_np[sample_idx]),
                        "v_pred_aux": float(aux_np[sample_idx]),
                        "raw_total_events": raw_total_events,
                        "normalized_total_events": normalized_total_events,
                        "source_scale": source_scale,
                        "is_eval": 1.0 if split == "eval" else 0.0,
                        "is_val": 1.0 if split == "val" else 0.0,
                        "is_train": 1.0 if split == "train" else 0.0,
                        "is_heldout": 1.0 if split in {"val", "eval"} else 0.0,
                    }
                    for cfg in SOURCE_CONFIGS:
                        name = source_name_from_path(cfg["source_path"])
                        row[f"is_{name}"] = 1.0 if source_name == name else 0.0
                    row.update(proxy_row)
                    row["raw_total_events"] = raw_total_events
                    row["normalized_total_events"] = normalized_total_events
                    row["source_scale"] = source_scale
                    rows.append(row)
                sample_offset += int(y_true.shape[0])
                progress.set_postfix(samples=len(rows))
            progress.close()

    csv_path = os.path.join(output_dir, "multiscattering_proxy_samples.csv")
    write_csv(csv_path, rows)

    correlations = {
        "abs_error": correlation_table(rows, "abs_error"),
        "residual": correlation_table(rows, "residual"),
        "is_val": correlation_table(rows, "is_val"),
        "is_eval": correlation_table(rows, "is_eval"),
        "is_heldout": correlation_table(rows, "is_heldout"),
    }
    source_indicator_corr = source_indicator_correlations(rows)
    scoped_correlations = build_scoped_correlations(rows)
    correlation_csv_path = os.path.join(output_dir, "multiscattering_proxy_correlations.csv")
    write_correlation_csv(correlation_csv_path, scoped_correlations)

    velocity_controlled_detail, velocity_controlled_summary = compute_velocity_controlled_correlations(rows)
    residualized_correlations = compute_residualized_correlations(rows)
    heldout_source_velocity_shift = compute_heldout_source_velocity_shift(rows)
    per_source = summarize_source(rows)
    source_velocity_summary = summarize_source_velocity(rows)
    top_val_bins = compute_top_problem_bins(source_velocity_summary, ["val"])
    top_eval_bins = compute_top_problem_bins(source_velocity_summary, ["eval"])
    top_heldout_bins = compute_top_problem_bins(source_velocity_summary, ["val", "eval"])

    velocity_controlled_path = os.path.join(output_dir, "velocity_controlled_correlations.csv")
    velocity_controlled_summary_path = os.path.join(output_dir, "velocity_controlled_correlation_summary.csv")
    residualized_path = os.path.join(output_dir, "residualized_correlations.csv")
    source_shift_path = os.path.join(output_dir, "heldout_source_velocity_shift.csv")
    source_velocity_summary_path = os.path.join(output_dir, "source_velocity_summary.csv")
    write_table_csv(velocity_controlled_path, velocity_controlled_detail)
    write_table_csv(velocity_controlled_summary_path, velocity_controlled_summary)
    write_table_csv(residualized_path, residualized_correlations)
    write_table_csv(source_shift_path, heldout_source_velocity_shift)
    write_table_csv(source_velocity_summary_path, source_velocity_summary)

    plot_paths = make_plots_v2(rows, output_dir, residualized_correlations, heldout_source_velocity_shift)

    report_path = os.path.join(output_dir, "multiscattering_proxy_report.md")
    write_report_v2(
        report_path,
        config,
        rows,
        correlations,
        source_indicator_corr,
        velocity_controlled_summary,
        residualized_correlations,
        per_source,
        source_velocity_summary,
        top_val_bins,
        top_eval_bins,
        top_heldout_bins,
        heldout_source_velocity_shift,
        plot_paths,
        time.time() - start_time,
    )

    print(f"Saved sample CSV: {csv_path}")
    print(f"Saved correlation CSV: {correlation_csv_path}")
    print(f"Saved velocity-controlled CSV: {velocity_controlled_path}")
    print(f"Saved residualized CSV: {residualized_path}")
    print(f"Saved source shift CSV: {source_shift_path}")
    print(f"Saved source velocity summary CSV: {source_velocity_summary_path}")
    print(f"Saved report: {report_path}")
    for name, path in plot_paths.items():
        print(f"Saved plot {name}: {path}")


if __name__ == "__main__":
    main()
