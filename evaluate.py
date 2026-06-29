import csv
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
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
from model import SNN_CNN_Hybrid
from train_cross import (
    build_prediction_records,
    compute_condition_velocity_metrics,
    compute_sub_condition_velocity_metrics,
    save_condition_velocity_metrics_csv,
    save_sub_condition_velocity_metrics_csv,
)


SAVE_GLOBAL_PREDICTION_MAPS = True
MAX_PREDICTION_MAP_PREVIEWS = 24


def load_trusted_checkpoint(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def unpack_eval_batch(batch):
    if len(batch) == 14:
        (
            x_seq_sparse_data,
            y_true,
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
        ) = batch
    elif len(batch) == 13:
        (
            x_seq_sparse_data,
            y_true,
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
        ) = batch
        metadata = None
    elif len(batch) == 11:
        (
            x_seq_sparse_data,
            y_true,
            d_values,
            env_maps,
            source_ids,
            K_max,
            beta_max,
            log_beta_max,
            condition,
            phantom_flag,
            metadata,
        ) = batch
        batch_size = int(y_true.shape[0])
        sub_condition = ["unknown"] * batch_size
        split_group = ["train_val"] * batch_size
        quality = ["legacy"] * batch_size
    elif len(batch) == 10:
        (
            x_seq_sparse_data,
            y_true,
            d_values,
            env_maps,
            source_ids,
            K_max,
            beta_max,
            log_beta_max,
            condition,
            phantom_flag,
        ) = batch
        metadata = None
        batch_size = int(y_true.shape[0])
        sub_condition = ["unknown"] * batch_size
        split_group = ["train_val"] * batch_size
        quality = ["legacy"] * batch_size
    else:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids = batch[:5]
        batch_size = int(y_true.shape[0])
        K_max = torch.ones(batch_size, dtype=torch.float32)
        beta_max = torch.ones(batch_size, dtype=torch.float32)
        log_beta_max = torch.zeros(batch_size, dtype=torch.float32)
        condition = ["unknown"] * batch_size
        sub_condition = ["unknown"] * batch_size
        split_group = ["train_val"] * batch_size
        quality = ["legacy"] * batch_size
        phantom_flag = torch.zeros(batch_size, dtype=torch.float32)
        metadata = batch[5] if len(batch) > 5 else None
    return x_seq_sparse_data, y_true, d_values, env_maps, source_ids, K_max, beta_max, log_beta_max, condition, sub_condition, split_group, quality, phantom_flag, metadata


def merge_checkpoint_kmax_config(test_data_config, checkpoint):
    if not isinstance(checkpoint, dict):
        return test_data_config
    checkpoint_data_config = checkpoint.get("data_config", {})
    candidate_maps = []
    if isinstance(checkpoint_data_config, dict):
        for value in checkpoint_data_config.values():
            if isinstance(value, dict):
                candidate_maps.append(value)
        candidate_maps.append(checkpoint_data_config)
    merged = {}
    for path, config_value in test_data_config.items():
        restored = None
        for mapping in candidate_maps:
            if path in mapping and isinstance(mapping[path], dict) and "K_max" in mapping[path]:
                restored = mapping[path]
                break
        if restored is not None and not isinstance(config_value, dict):
            merged[path] = {
                "d_value": float(config_value),
                "K_max": restored.get("K_max", 1.0),
                "condition": restored.get("condition", "unknown"),
                "sub_condition": restored.get("sub_condition", "unknown"),
                "phantom_flag": restored.get("phantom_flag", -1),
                "split_group": restored.get("split_group", "evaluate"),
                "quality": restored.get("quality", "legacy"),
                "use_for_training": restored.get("use_for_training", False),
            }
        else:
            merged[path] = config_value
    return merged


def compute_scalar_metrics(v_true_list, v_pred_list):
    v_true = np.asarray(v_true_list, dtype=np.float64)
    v_pred = np.asarray(v_pred_list, dtype=np.float64)
    finite = np.isfinite(v_true) & np.isfinite(v_pred)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan")
    v_true = v_true[finite]
    v_pred = v_pred[finite]
    abs_err = np.abs(v_true - v_pred)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))
    mape = float(np.mean(abs_err / np.maximum(np.abs(v_true), 1e-8)) * 100.0)
    return mae, rmse, mape


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or y.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_rank_accuracy(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(pred) & np.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    if pred.size < 2 or target.size < 2:
        return float("nan")
    target_diff = target.reshape(-1, 1) - target.reshape(1, -1)
    pred_diff = pred.reshape(-1, 1) - pred.reshape(1, -1)
    valid = target_diff > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(pred_diff[valid] > 0))


def format_duration(elapsed_seconds):
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def safe_spearman(x, y):
    return safe_pearson(_rankdata(x), _rankdata(y))


def compute_diagnostic_correlations(v_true, v_pred, raw_total_events):
    abs_err = np.abs(np.asarray(v_true, dtype=np.float64) - np.asarray(v_pred, dtype=np.float64))
    return {
        "pred_vs_raw_total_events_pearson": safe_pearson(v_pred, raw_total_events),
        "pred_vs_raw_total_events_spearman": safe_spearman(v_pred, raw_total_events),
        "label_vs_raw_total_events_pearson": safe_pearson(v_true, raw_total_events),
        "abs_error_vs_raw_total_events_pearson": safe_pearson(abs_err, raw_total_events),
        "abs_error_vs_raw_total_events_spearman": safe_spearman(abs_err, raw_total_events),
    }


def compute_mae_by_velocity(v_true, v_pred):
    rows = []
    v_true = np.asarray(v_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    for velocity in sorted(set(v_true.tolist())):
        mask = v_true == velocity
        pred_values = v_pred[mask]
        mae, rmse, mape = compute_scalar_metrics(v_true[mask], v_pred[mask])
        pred_mean = float(pred_values.mean()) if pred_values.size else float("nan")
        rows.append(
            {
                "velocity": velocity,
                "samples": int(mask.sum()),
                "pred_mean": pred_mean,
                "pred_std": float(pred_values.std()) if pred_values.size else float("nan"),
                "bias": pred_mean - float(velocity),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )
    return rows


def format_markdown_table(title, mapping, key_name):
    lines = [f"### {title}", "", f"| {key_name} | Samples |", "| --- | ---: |"]
    if not mapping:
        lines.append("| (empty) | 0 |")
    else:
        for key, value in mapping.items():
            key_text = f"{key:.6f}" if isinstance(key, float) else str(key)
            lines.append(f"| `{key_text}` | {value} |")
    lines.append("")
    return "\n".join(lines)


def flatten_sampler_metadata(dataset, plan, seed, max_batches):
    sampler = SourceVelocityBatchSampler(dataset.source_velocity_sample_indices, plan, seed=seed)
    ordered = []
    for batch_idx, batch in enumerate(sampler):
        if batch_idx >= max_batches:
            break
        ordered.extend(dataset.sample_metadata[index] for index in batch)
    return ordered


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
    if effective_batches == 0:
        raise ValueError(f"{split_name} max_batches={max_batches} is too small for balanced velocity batches.")
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


def build_source_velocity_loader(dataset, batch_size, num_workers, collate_fn, max_batches, split_name):
    plan = compute_source_velocity_sampling_plan(dataset, batch_size, max_batches, split_name)
    sampler = SourceVelocityBatchSampler(dataset.source_velocity_sample_indices, plan, seed=20260428)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers), plan


def write_evaluation_report(report_path, run_info, eval_record):
    test_ds = run_info["test_ds"]
    eval_sampling_plan = run_info["eval_sampling_plan"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Run timestamp: `{run_info['timestamp']}`",
        f"- Status: `{run_info['status']}`",
        f"- Device: `{run_info['device']}`",
        f"- Duration: `{format_duration(run_info['elapsed'])}`",
        f"- Model weights path: `{run_info['model_weights_path']}`",
        f"- Checkpoint stage: `{run_info['checkpoint_stage']}`",
        f"- Generalization output dir: `{run_info['generalization_output_dir']}`",
        f"- Plot path: `{run_info['save_plot_path']}`",
        f"- Prediction CSV path: `{run_info['save_prediction_path']}`",
        "",
        "## Run Config",
        "",
        f"- window_ms: `{run_info['window_ms']}`",
        f"- base_dt_us: `{run_info['base_dt_us']}`",
        f"- base_total_steps: `{run_info['base_total_steps']}`",
        f"- base_block_size: `{run_info['base_block_size']}`",
        f"- snn_bin_size: `{run_info['snn_bin_size']}`",
        f"- snn_step_us: `{run_info['snn_step_us']}`",
        f"- snn_steps: `{run_info['snn_steps']}`",
        f"- snn_input_scale_mode: `{run_info['snn_input_scale_mode']}`",
        f"- batch_size: `{run_info['batch_size']}`",
        f"- dt_us: `{run_info['dt_us']}`",
        f"- spatial_shape: `{run_info['spatial_shape']}`",
        f"- patch_shape: `{run_info['patch_shape']}`",
        f"- max_eval_batches: `{run_info['max_eval_batches']}`",
        f"- max_velocity: `{run_info['max_velocity']}`",
        f"- use_beta_conditioning: `{run_info.get('use_beta_conditioning', False)}`",
        f"- use_bounded_scatter: `{run_info.get('use_bounded_scatter', False)}`",
        f"- scatter_scale: `{run_info.get('scatter_scale', float('nan'))}`",
        "",
        "### Batch Limit Summary",
        "",
        "| Split | Mode | Requested Max Batches | Effective Max Batches | Sources | Velocities Per Batch | Batches Per Source | Samples Per Source | Adjusted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| Eval | `{eval_sampling_plan['sampling_mode']}` | "
        f"{eval_sampling_plan['requested_batches'] if eval_sampling_plan['requested_batches'] is not None else 'all'} | "
        f"{eval_sampling_plan['effective_batches']} | {eval_sampling_plan['num_sources']} | "
        f"{eval_sampling_plan['velocities_per_batch']} | {eval_sampling_plan['batches_per_source']} | "
        f"{eval_sampling_plan['samples_per_source']} | {'yes' if eval_sampling_plan['was_adjusted'] else 'no'} |",
        "",
        "## Dataset Summary",
        "",
        f"- eval_samples: `{len(test_ds)}`",
        f"- eval_batches: `{run_info['eval_batches']}`",
        "",
        format_markdown_table("Eval Samples Per Source", test_ds.source_sample_counts, "Source"),
        format_markdown_table("Eval Samples Per Velocity", test_ds.velocity_sample_counts, "Velocity"),
        "## Evaluation Metrics",
        "",
        "| Eval Batches | Eval Samples | Final MAE | Final RMSE | Final MAPE | Final Pred Std | Final Pred Range | Final Pred/Label Pearson | Final Rank Acc | Clipped MAE | Clipped RMSE | Clipped MAPE | Clipped Pred Range | Aux V MAE | Aux V Std | Aux V Range | Tau Sample Range | Log Tau Range |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    if eval_record is None:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        lines.append(
            f"| {eval_record['eval_batches_processed']}/{eval_record['eval_batches_available']} | "
            f"{eval_record['eval_samples_seen']}/{eval_record['eval_samples_available']} | "
            f"{eval_record['mae']:.6f} | {eval_record['rmse']:.6f} | {eval_record['mape']:.2f}% | "
            f"{eval_record['pred_std']:.6f} | {eval_record['pred_min']:.6f}-{eval_record['pred_max']:.6f} | "
            f"{eval_record['pred_label_pearson']:.6f} | {eval_record['rank_accuracy']:.6f} | "
            f"{eval_record['clipped_mae']:.6f} | {eval_record['clipped_rmse']:.6f} | "
            f"{eval_record['clipped_mape']:.2f}% | "
            f"{eval_record['clipped_pred_min']:.6f}-{eval_record['clipped_pred_max']:.6f} | "
            f"{eval_record['aux_v_mae']:.6f} | {eval_record['aux_v_std']:.6f} | "
            f"{eval_record['aux_v_min']:.6f}-{eval_record['aux_v_max']:.6f} | "
            f"{eval_record['tau_sample_min']:.6e}-{eval_record['tau_sample_max']:.6e} | "
            f"{eval_record['log_tau_min']:.6f}-{eval_record['log_tau_max']:.6f} |"
        )
    if eval_record is not None:
        lines.extend(["", "### Raw Event Correlations", "", "| Metric | Value |", "| --- | ---: |"])
        for key, value in eval_record.get("correlations", {}).items():
            lines.append(f"| `{key}` | {value:.6f} |")
        lines.extend(
            [
                "",
                "### Beta/Scatter Diagnostics",
                "",
                f"- beta_eff_lt_beta_max: `{eval_record.get('beta_eff_lt_beta_max')}`",
                f"- gamma_mean/std: `{eval_record.get('gamma_mean', float('nan')):.6f}` / `{eval_record.get('gamma_std', float('nan')):.6f}`",
                f"- beta_eff_ratio_mean: `{eval_record.get('beta_eff_ratio_mean', float('nan')):.6f}`",
                f"- scatter_delta_mean: `{eval_record.get('scatter_delta_mean', float('nan')):.6f}`",
                f"- corr_gamma_velocity: `{eval_record.get('corr_gamma_velocity', float('nan')):.6f}`",
                f"- corr_scatter_delta_velocity: `{eval_record.get('corr_scatter_delta_velocity', float('nan')):.6f}`",
            ]
        )
        lines.extend(
            [
                "",
                "### MAE By Velocity",
                "",
                "| Velocity | Samples | Pred Mean | Pred Std | Bias | MAE | RMSE | MAPE |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in eval_record.get("mae_by_velocity", []):
            lines.append(
                f"| {row['velocity']:.6f} | {row['samples']} | "
                f"{row['pred_mean']:.6f} | {row['pred_std']:.6f} | {row['bias']:.6f} | "
                f"{row['mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Global Prediction Maps",
            "",
            "- These maps render the sample-level global prediction back onto the ROI for quick visual inspection.",
            "- They are not local velocity-field predictions; every valid pixel in a sample mask receives the same `v_final` value.",
            f"- Prediction map dir: `{run_info.get('prediction_map_outputs', {}).get('prediction_map_dir', '')}`",
            f"- Mean prediction map PNG: `{run_info.get('prediction_map_outputs', {}).get('mean_prediction_map_png', '')}`",
            f"- Mean prediction map NPY: `{run_info.get('prediction_map_outputs', {}).get('mean_prediction_map_npy', '')}`",
            f"- Sample-count map NPY: `{run_info.get('prediction_map_outputs', {}).get('sample_count_map_npy', '')}`",
            f"- Manifest CSV: `{run_info.get('prediction_map_outputs', {}).get('manifest_csv', '')}`",
            f"- Sample map files: `{run_info.get('prediction_map_outputs', {}).get('num_sample_maps', 0)}`",
            f"- Preview PNG files: `{run_info.get('prediction_map_outputs', {}).get('num_preview_png', 0)}`",
            "",
            "## Notes",
            "",
            "- Final prediction is `d_values / tau_pred`; `tau_pred` denotes `tau_eff` when bounded scatter is enabled.",
            "- `v_pred` is auxiliary only.",
            "- beta_eff is constrained as sigmoid(raw_gamma) * beta_max.",
            "",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _safe_filename(text):
    text = str(text or "unknown")
    keep = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def _source_mask_for_prediction_map(dataset, source_path, spatial_shape):
    source_masks = getattr(dataset, "source_channel_masks", {})
    mask = source_masks.get(source_path)
    if mask is None:
        return np.ones(spatial_shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(spatial_shape):
        raise ValueError(
            f"channel mask shape mismatch while rendering prediction map for source `{source_path}`: "
            f"expected {spatial_shape}, got {mask.shape}."
        )
    return mask


def save_prediction_map_png(path, pred_map, title, vmin=None, vmax=None):
    plt.figure(figsize=(14, 3.2))
    masked = np.ma.masked_invalid(pred_map)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0.0)
    im = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
    plt.colorbar(im, label="Predicted velocity (mm/s)")
    plt.xlabel("Local col in ROI")
    plt.ylabel("Local row in ROI")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_global_prediction_maps(
    output_dir,
    dataset,
    metadata_list,
    pred_values,
    true_values,
    condition_values,
    sub_condition_values,
    spatial_shape,
    max_preview=24,
):
    os.makedirs(output_dir, exist_ok=True)
    pred_values = np.asarray(pred_values, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float64)
    accum = np.zeros(spatial_shape, dtype=np.float64)
    counts = np.zeros(spatial_shape, dtype=np.float64)
    manifest_rows = []

    finite_preds = pred_values[np.isfinite(pred_values)]
    vmin = float(np.nanpercentile(finite_preds, 5)) if finite_preds.size else None
    vmax = float(np.nanpercentile(finite_preds, 95)) if finite_preds.size else None
    if vmin is not None and vmax is not None and abs(vmax - vmin) < 1e-12:
        vmin = float(finite_preds.min())
        vmax = float(finite_preds.max() + 1e-6)

    for idx, meta in enumerate(metadata_list):
        source_path = meta.get("source_path", "")
        pred_value = float(pred_values[idx])
        source_mask = _source_mask_for_prediction_map(dataset, source_path, spatial_shape)
        if np.isfinite(pred_value):
            accum[source_mask] += pred_value
            counts[source_mask] += 1.0

        pred_map = np.full(spatial_shape, np.nan, dtype=np.float32)
        if np.isfinite(pred_value):
            pred_map[source_mask] = pred_value

        source_stem = _safe_filename(os.path.basename(str(source_path).rstrip("/\\")) or source_path)
        file_stem = _safe_filename(os.path.splitext(os.path.basename(str(meta.get("file_path", ""))))[0])
        map_name = f"sample_{idx:04d}_{source_stem}_{file_stem}_global_prediction_map"
        npy_path = os.path.join(output_dir, f"{map_name}.npy")
        png_path = os.path.join(output_dir, f"{map_name}.png")
        np.save(npy_path, pred_map)
        if idx < max_preview:
            title = (
                "Global prediction rendered as ROI map | "
                f"pred={pred_value:.4f} mm/s, label={true_values[idx]:.4f}"
            )
            save_prediction_map_png(png_path, pred_map, title, vmin=vmin, vmax=vmax)
        else:
            png_path = ""

        manifest_rows.append(
            {
                "idx": idx,
                "source_path": source_path,
                "file_path": meta.get("file_path", ""),
                "condition": condition_values[idx] if idx < len(condition_values) else "unknown",
                "sub_condition": sub_condition_values[idx] if idx < len(sub_condition_values) else "unknown",
                "true_velocity": float(true_values[idx]) if idx < len(true_values) else float("nan"),
                "pred_velocity": pred_value,
                "mask_area_pixels": int(source_mask.sum()),
                "prediction_map_npy": npy_path,
                "prediction_map_png": png_path,
            }
        )

    mean_map = np.full(spatial_shape, np.nan, dtype=np.float32)
    valid = counts > 0
    mean_map[valid] = (accum[valid] / counts[valid]).astype(np.float32)
    mean_npy_path = os.path.join(output_dir, "evaluate_global_prediction_map_mean.npy")
    mean_png_path = os.path.join(output_dir, "evaluate_global_prediction_map_mean.png")
    count_npy_path = os.path.join(output_dir, "evaluate_global_prediction_map_sample_count.npy")
    manifest_path = os.path.join(output_dir, "evaluate_global_prediction_map_manifest.csv")
    np.save(mean_npy_path, mean_map)
    np.save(count_npy_path, counts.astype(np.float32))
    save_prediction_map_png(
        mean_png_path,
        mean_map,
        "Mean global prediction rendered as ROI map",
        vmin=vmin,
        vmax=vmax,
    )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "idx",
            "source_path",
            "file_path",
            "condition",
            "sub_condition",
            "true_velocity",
            "pred_velocity",
            "mask_area_pixels",
            "prediction_map_npy",
            "prediction_map_png",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    return {
        "prediction_map_dir": output_dir,
        "mean_prediction_map_npy": mean_npy_path,
        "mean_prediction_map_png": mean_png_path,
        "sample_count_map_npy": count_npy_path,
        "manifest_csv": manifest_path,
        "num_sample_maps": len(manifest_rows),
        "num_preview_png": min(len(manifest_rows), max_preview),
    }


def evaluate_generalization():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dt_us = 20
    window_ms = 200
    base_total_steps = int(window_ms * 1000 / base_dt_us)
    base_block_size = 400
    snn_bin_size = 40
    snn_step_us = base_dt_us * snn_bin_size
    snn_steps = base_total_steps // snn_bin_size
    snn_input_scale_mode = "sqrt"
    batch_size = 2
    num_workers = 0
    spatial_shape = (100, 1200)
    patch_shape = (50, 50)
    dt_us = base_dt_us
    max_velocity = 2.0
    max_eval_batches = None
    event_norm_mode = "source_scale"
    event_norm_clip = (0.25, 4.0)

    test_data_config = {
        "/data/zm/Weiliukong/6.24/withf2": 0.023138,
    }
    # Dataset `mask_path` is the full-frame hot-pixel mask only, shape (800, 1280).
    # Per-source channel masks are read from data_config/checkpoint via `channel_mask_path`.
    mask_path = "/data/zm/Weiliukong/6.17/mask/blood_maskweiliukong_hot_pixel_mask.npy"
    model_weights_path = "/data/zm/Weiliukong/6.24/Train_result/model/best_blood_flow_model.pth"
    loss_curve_dir = "/data/zm/Weiliukong/6.24/Train_result/evaluate/no6/Loss_curve"
    report_dir = "/data/zm/Weiliukong/6.24/Train_result/evaluate/no6/Markdown"
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generalization_output_dir = os.path.join(loss_curve_dir, f"generalization_{report_timestamp}")
    save_plot_path = os.path.join(generalization_output_dir, "generalization_evaluate.png")
    save_prediction_path = os.path.join(generalization_output_dir, "evaluate_predictions.csv")
    save_condition_velocity_metrics_path = os.path.join(generalization_output_dir, "evaluate_condition_velocity_metrics.csv")
    save_sub_condition_velocity_metrics_path = os.path.join(generalization_output_dir, "evaluate_sub_condition_velocity_metrics.csv")
    prediction_map_dir = os.path.join(generalization_output_dir, "global_prediction_maps")
    report_path = os.path.join(report_dir, f"evaluate_{report_timestamp}.md")

    os.makedirs(generalization_output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print(f"=> Loading model weights from: {model_weights_path}")
    checkpoint = load_trusted_checkpoint(model_weights_path, map_location=device)
    checkpoint_has_event_norm_stats = isinstance(checkpoint, dict) and "event_norm_stats" in checkpoint
    if isinstance(checkpoint, dict) and "input_config" in checkpoint:
        input_config = checkpoint["input_config"]
        window_ms = int(input_config.get("window_ms", window_ms))
        base_dt_us = int(input_config.get("base_dt_us", base_dt_us))
        base_total_steps = int(input_config.get("base_total_steps", base_total_steps))
        base_block_size = int(input_config.get("base_block_size", base_block_size))
        snn_bin_size = int(input_config.get("snn_bin_size", snn_bin_size))
        snn_step_us = int(input_config.get("snn_step_us", base_dt_us * snn_bin_size))
        snn_steps = int(input_config.get("snn_steps", base_total_steps // snn_bin_size))
        snn_input_scale_mode = input_config.get("snn_input_scale_mode", snn_input_scale_mode)
        dt_us = base_dt_us
    checkpoint_event_norm_stats = checkpoint.get("event_norm_stats") if checkpoint_has_event_norm_stats else None
    checkpoint_reference_mean = (
        checkpoint_event_norm_stats.get("reference_mean_events_per_sample")
        if checkpoint_event_norm_stats is not None
        else None
    )
    if not checkpoint_has_event_norm_stats or checkpoint_reference_mean is None:
        print("WARNING: checkpoint has no usable event_norm_stats; falling back to event_norm_mode='none'.")
        event_norm_mode = "none"

    has_beta_conditioning_config = isinstance(checkpoint, dict) and "beta_conditioning_config" in checkpoint
    beta_conditioning_config = checkpoint.get("beta_conditioning_config", {}) if has_beta_conditioning_config else {}
    use_beta_conditioning = bool(beta_conditioning_config.get("use_beta_conditioning", has_beta_conditioning_config))
    use_bounded_scatter = bool(beta_conditioning_config.get("use_bounded_scatter", has_beta_conditioning_config))
    scatter_scale = float(beta_conditioning_config.get("scatter_scale", 0.3))
    if isinstance(checkpoint, dict):
        checkpoint_data_config = checkpoint.get("data_config", {})
        if isinstance(checkpoint_data_config, dict) and isinstance(checkpoint_data_config.get("evaluate"), dict) and checkpoint_data_config.get("evaluate"):
            test_data_config = checkpoint_data_config["evaluate"]
    test_data_config = merge_checkpoint_kmax_config(test_data_config, checkpoint)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_stage = checkpoint.get("stage", "unknown")
        checkpoint_event_norm_config = checkpoint.get("event_norm_config", {})
        max_velocity = checkpoint_event_norm_config.get("max_velocity", checkpoint.get("max_velocity", max_velocity))
        model = SNN_CNN_Hybrid(
            in_channels=1,
            max_velocity=max_velocity,
            use_beta_conditioning=use_beta_conditioning,
            use_bounded_scatter=use_bounded_scatter,
            scatter_scale=scatter_scale,
        ).to(device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"WARNING: checkpoint loaded with strict=False; missing={missing}, unexpected={unexpected}")
    else:
        checkpoint_stage = "legacy_state_dict"
        checkpoint_event_norm_config = {}
        model = SNN_CNN_Hybrid(
            in_channels=1,
            max_velocity=max_velocity,
            use_beta_conditioning=use_beta_conditioning,
            use_bounded_scatter=use_bounded_scatter,
            scatter_scale=scatter_scale,
        ).to(device)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            print(f"WARNING: legacy checkpoint loaded with strict=False; missing={missing}, unexpected={unexpected}")
    model.eval()
    print(
        f"=> Loaded checkpoint metadata | stage={checkpoint_stage}, "
        f"event_norm_mode={event_norm_mode}, max_velocity={max_velocity}, "
        f"use_beta_conditioning={use_beta_conditioning}, use_bounded_scatter={use_bounded_scatter}, "
        f"scatter_scale={scatter_scale}"
    )

    print("\n=> Loading test dataset...")
    test_dataset = FlexibleBloodFlowDataset(
        data_config=test_data_config,
        mask_path=mask_path,
        T=1,
        seq_len=base_total_steps,
        dt_us=dt_us,
        max_velocity=max_velocity,
        event_norm_mode=event_norm_mode,
        event_norm_stats=checkpoint_event_norm_stats,
        event_norm_reference_mean=checkpoint_reference_mean,
        event_norm_clip=event_norm_clip,
        event_intensity_jitter_range=None,
    )
    if len(test_dataset) == 0:
        print("No valid evaluation samples were built.")
        return

    test_loader, eval_sampling_plan = build_source_velocity_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=sequence_sparse_collate,
        max_batches=max_eval_batches,
        split_name="Eval",
    )
    effective_max_eval_batches = eval_sampling_plan["effective_batches"]
    eval_order_metadata = flatten_sampler_metadata(
        test_dataset,
        eval_sampling_plan,
        seed=20260428,
        max_batches=effective_max_eval_batches,
    )

    all_v_true = []
    all_v_final = []
    all_v_aux = []
    all_tau_pred = []
    all_log_tau = []
    all_K_max = []
    all_beta_max = []
    all_gamma = []
    all_beta_eff = []
    all_beta_eff_ratio = []
    all_scatter_delta = []
    all_tau_base = []
    all_tau_eff = []
    all_log_tau_base = []
    all_log_tau_eff = []
    all_condition = []
    all_sub_condition = []
    all_split_group = []
    all_quality = []
    all_phantom_flag = []
    processed_batches = 0

    print(f"=> Start evaluation on {len(test_dataset)} samples...")
    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(test_loader),
            total=effective_max_eval_batches,
            desc="Evaluate [Eval]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, batch in progress_bar:
            if batch_idx >= effective_max_eval_batches:
                break
            (
                x_seq_sparse_data,
                y_true,
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
            ) = unpack_eval_batch(batch)
            d_values = d_values.to(device)
            beta_max = beta_max.to(device)
            log_beta_max = log_beta_max.to(device)
            manager = DenseBlockManager(
                x_seq_sparse_data,
                batch_size=y_true.shape[0],
                spatial_shape=spatial_shape,
                patch_shape=patch_shape,
            )
            model_output = model(
                dataloader_or_generator=manager,
                base_total_steps=base_total_steps,
                base_block_size=base_block_size,
                snn_bin_size=snn_bin_size,
                snn_input_scale_mode=snn_input_scale_mode,
                base_dt_us=base_dt_us,
                beta_max=beta_max,
                log_beta_max=log_beta_max,
            )
            v_aux = model_output["v_pred"]
            v_final = d_values / torch.clamp(model_output["tau_pred"], min=1e-8)

            all_v_true.extend(y_true.numpy().tolist())
            all_v_final.extend(v_final.cpu().numpy().tolist())
            all_v_aux.extend(v_aux.cpu().numpy().tolist())
            all_tau_pred.extend(model_output["tau_pred"].cpu().numpy().tolist())
            all_log_tau.extend(model_output["log_tau_pred"].cpu().numpy().tolist())
            all_K_max.extend(K_max.cpu().numpy().tolist())
            all_beta_max.extend(model_output["beta_max"].cpu().numpy().tolist())
            all_gamma.extend(model_output["gamma"].cpu().numpy().tolist())
            all_beta_eff.extend(model_output["beta_eff"].cpu().numpy().tolist())
            all_beta_eff_ratio.extend(model_output["beta_eff_ratio"].cpu().numpy().tolist())
            all_scatter_delta.extend(model_output["scatter_delta"].cpu().numpy().tolist())
            all_tau_base.extend(model_output["tau_base"].cpu().numpy().tolist())
            all_tau_eff.extend(model_output["tau_eff"].cpu().numpy().tolist())
            all_log_tau_base.extend(model_output["log_tau_base"].cpu().numpy().tolist())
            all_log_tau_eff.extend(model_output["log_tau_eff"].cpu().numpy().tolist())
            all_condition.extend(condition)
            all_sub_condition.extend(sub_condition)
            all_split_group.extend(split_group)
            all_quality.extend(quality)
            all_phantom_flag.extend(phantom_flag.cpu().numpy().tolist())
            processed_batches += 1

            progress_bar.set_postfix(
                final=f"{v_final.min().item():.3f}-{v_final.max().item():.3f}",
                final_std=f"{v_final.std(unbiased=False).item():.3e}",
            )
        progress_bar.close()

    if not all_v_true:
        print("No evaluation batches were processed.")
        return

    v_true_arr = np.asarray(all_v_true, dtype=np.float64)
    v_final_arr = np.asarray(all_v_final, dtype=np.float64)
    v_aux_arr = np.asarray(all_v_aux, dtype=np.float64)
    tau_arr = np.asarray(all_tau_pred, dtype=np.float64)
    log_tau_arr = np.asarray(all_log_tau, dtype=np.float64)
    K_max_arr = np.asarray(all_K_max, dtype=np.float64)
    beta_max_arr = np.asarray(all_beta_max, dtype=np.float64)
    gamma_arr = np.asarray(all_gamma, dtype=np.float64)
    beta_eff_arr = np.asarray(all_beta_eff, dtype=np.float64)
    beta_eff_ratio_arr = np.asarray(all_beta_eff_ratio, dtype=np.float64)
    scatter_delta_arr = np.asarray(all_scatter_delta, dtype=np.float64)
    tau_base_arr = np.asarray(all_tau_base, dtype=np.float64)
    tau_eff_arr = np.asarray(all_tau_eff, dtype=np.float64)
    log_tau_base_arr = np.asarray(all_log_tau_base, dtype=np.float64)
    log_tau_eff_arr = np.asarray(all_log_tau_eff, dtype=np.float64)
    phantom_flag_arr = np.asarray(all_phantom_flag, dtype=np.float64)
    eval_order_metadata = eval_order_metadata[:len(v_true_arr)]
    raw_total_events_arr = np.asarray(
        [meta.get("raw_total_events", np.nan) for meta in eval_order_metadata],
        dtype=np.float64,
    )

    sorted_idx = np.argsort(v_true_arr)
    sorted_v_true = v_true_arr[sorted_idx]
    sorted_v_pred = v_final_arr[sorted_idx]
    sorted_v_pred_clipped = np.clip(sorted_v_pred, 0.0, max_velocity)
    sorted_v_aux = v_aux_arr[sorted_idx]
    sorted_tau = tau_arr[sorted_idx]
    sorted_log_tau = log_tau_arr[sorted_idx]
    sorted_K_max = K_max_arr[sorted_idx]
    sorted_beta_max = beta_max_arr[sorted_idx]
    sorted_gamma = gamma_arr[sorted_idx]
    sorted_beta_eff = beta_eff_arr[sorted_idx]
    sorted_beta_eff_ratio = beta_eff_ratio_arr[sorted_idx]
    sorted_scatter_delta = scatter_delta_arr[sorted_idx]
    sorted_tau_base = tau_base_arr[sorted_idx]
    sorted_tau_eff = tau_eff_arr[sorted_idx]
    sorted_log_tau_base = log_tau_base_arr[sorted_idx]
    sorted_log_tau_eff = log_tau_eff_arr[sorted_idx]
    sorted_condition = [all_condition[i] for i in sorted_idx]
    sorted_sub_condition = [all_sub_condition[i] for i in sorted_idx]
    sorted_split_group = [all_split_group[i] for i in sorted_idx]
    sorted_quality = [all_quality[i] for i in sorted_idx]
    sorted_phantom_flag = phantom_flag_arr[sorted_idx]
    sorted_metadata = [eval_order_metadata[i] for i in sorted_idx]
    sorted_raw_total_events = raw_total_events_arr[sorted_idx]

    abs_err = np.abs(sorted_v_true - sorted_v_pred)
    clipped_abs_err = np.abs(sorted_v_true - sorted_v_pred_clipped)
    rel_err_pct = abs_err / np.maximum(np.abs(sorted_v_true), 1e-8) * 100.0
    mae, rmse, mape = compute_scalar_metrics(sorted_v_true, sorted_v_pred)
    clipped_mae, clipped_rmse, clipped_mape = compute_scalar_metrics(sorted_v_true, sorted_v_pred_clipped)
    aux_mae, _, _ = compute_scalar_metrics(sorted_v_true, sorted_v_aux)
    correlations = compute_diagnostic_correlations(sorted_v_true, sorted_v_pred, sorted_raw_total_events)
    mae_by_velocity = compute_mae_by_velocity(sorted_v_true, sorted_v_pred)

    eval_record = {
        "eval_batches_processed": processed_batches,
        "eval_batches_available": len(test_loader),
        "eval_samples_seen": len(sorted_v_true),
        "eval_samples_available": len(test_dataset),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pred_std": float(sorted_v_pred.std()) if sorted_v_pred.size else float("nan"),
        "pred_min": float(sorted_v_pred.min()) if sorted_v_pred.size else float("nan"),
        "pred_max": float(sorted_v_pred.max()) if sorted_v_pred.size else float("nan"),
        "pred_label_pearson": safe_pearson(sorted_v_pred, sorted_v_true),
        "rank_accuracy": pairwise_rank_accuracy(sorted_v_pred, sorted_v_true),
        "clipped_mae": clipped_mae,
        "clipped_rmse": clipped_rmse,
        "clipped_mape": clipped_mape,
        "clipped_pred_std": float(sorted_v_pred_clipped.std()) if sorted_v_pred_clipped.size else float("nan"),
        "clipped_pred_min": float(sorted_v_pred_clipped.min()) if sorted_v_pred_clipped.size else float("nan"),
        "clipped_pred_max": float(sorted_v_pred_clipped.max()) if sorted_v_pred_clipped.size else float("nan"),
        "aux_v_mae": aux_mae,
        "aux_v_std": float(sorted_v_aux.std()) if sorted_v_aux.size else float("nan"),
        "aux_v_min": float(sorted_v_aux.min()) if sorted_v_aux.size else float("nan"),
        "aux_v_max": float(sorted_v_aux.max()) if sorted_v_aux.size else float("nan"),
        "tau_sample_min": float(sorted_tau.min()) if sorted_tau.size else float("nan"),
        "tau_sample_max": float(sorted_tau.max()) if sorted_tau.size else float("nan"),
        "log_tau_min": float(sorted_log_tau.min()) if sorted_log_tau.size else float("nan"),
        "log_tau_max": float(sorted_log_tau.max()) if sorted_log_tau.size else float("nan"),
        "gamma_mean": float(sorted_gamma.mean()) if sorted_gamma.size else float("nan"),
        "gamma_std": float(sorted_gamma.std()) if sorted_gamma.size else float("nan"),
        "beta_eff_ratio_mean": float(sorted_beta_eff_ratio.mean()) if sorted_beta_eff_ratio.size else float("nan"),
        "scatter_delta_mean": float(sorted_scatter_delta.mean()) if sorted_scatter_delta.size else float("nan"),
        "corr_gamma_velocity": safe_pearson(sorted_gamma, sorted_v_true),
        "corr_scatter_delta_velocity": safe_pearson(sorted_scatter_delta, sorted_v_true),
        "beta_eff_lt_beta_max": bool(sorted_beta_eff.size > 0 and np.all(sorted_beta_eff < sorted_beta_max + 1e-12)),
        "correlations": correlations,
        "mae_by_velocity": mae_by_velocity,
    }

    print("\n" + "=" * 60)
    print("=> Generalization Evaluation Report")
    print(f"=> Test sample count: {len(sorted_v_true)}")
    print(f"=> Eval batches: {processed_batches}/{len(test_loader)}")
    print(f"=> MAE:  {mae:.6f} mm/s")
    print(f"=> RMSE: {rmse:.6f} mm/s")
    print(f"=> MAPE: {mape:.2f} %")
    print(f"=> Pred std: {eval_record['pred_std']:.6f}")
    print(f"=> Pred range: [{eval_record['pred_min']:.6f}, {eval_record['pred_max']:.6f}]")
    print(f"=> Rank acc: {eval_record['rank_accuracy']:.6f}")
    print(f"=> Clipped MAE: {clipped_mae:.6f} mm/s")
    print(f"=> Aux v MAE: {aux_mae:.6f} mm/s")
    print("=" * 60 + "\n")

    with open(save_prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "true_velocity_mm_per_s",
                "d_value",
                "pred_velocity_mm_per_s",
                "pred_velocity_clipped_mm_per_s",
                "aux_velocity_mm_per_s",
                "tau_sample_s",
                "log_tau",
                "K_max",
                "beta_max",
                "gamma",
                "beta_eff",
                "beta_eff_ratio",
                "scatter_delta",
                "tau_base",
                "tau_eff",
                "log_tau_base",
                "log_tau_eff",
                "condition",
                "sub_condition",
                "split_group",
                "quality",
                "phantom_flag",
                "abs_error_mm_per_s",
                "clipped_abs_error_mm_per_s",
                "rel_error_pct",
                "raw_total_events",
                "normalized_total_events",
                "seq_start_idx",
                "source_path",
                "file_path",
            ]
        )
        for row_idx, meta in enumerate(sorted_metadata):
            writer.writerow(
                [
                    row_idx,
                    sorted_v_true[row_idx],
                    meta.get("d_val", np.nan),
                    sorted_v_pred[row_idx],
                    sorted_v_pred_clipped[row_idx],
                    sorted_v_aux[row_idx],
                    sorted_tau[row_idx],
                    sorted_log_tau[row_idx],
                    sorted_K_max[row_idx],
                    sorted_beta_max[row_idx],
                    sorted_gamma[row_idx],
                    sorted_beta_eff[row_idx],
                    sorted_beta_eff_ratio[row_idx],
                    sorted_scatter_delta[row_idx],
                    sorted_tau_base[row_idx],
                    sorted_tau_eff[row_idx],
                    sorted_log_tau_base[row_idx],
                    sorted_log_tau_eff[row_idx],
                    sorted_condition[row_idx],
                    sorted_sub_condition[row_idx],
                    sorted_split_group[row_idx],
                    sorted_quality[row_idx],
                    sorted_phantom_flag[row_idx],
                    abs_err[row_idx],
                    clipped_abs_err[row_idx],
                    rel_err_pct[row_idx],
                    sorted_raw_total_events[row_idx],
                    meta.get("normalized_total_events_est", np.nan),
                    meta.get("seq_start_idx", ""),
                    meta.get("source_path", ""),
                    meta.get("file_path", ""),
                ]
            )

    prediction_records = build_prediction_records(
        {
            "epoch_idx": checkpoint.get("epoch", float("nan")) if isinstance(checkpoint, dict) else float("nan"),
            "split_name": "Evaluate",
            "v_true": sorted_v_true.tolist(),
            "v_pred": sorted_v_pred.tolist(),
            "v_aux": sorted_v_aux.tolist(),
            "d_values": [meta.get("d_val", np.nan) for meta in sorted_metadata],
            "tau_pred_values": sorted_tau.tolist(),
            "log_tau_values": sorted_log_tau.tolist(),
            "K_max_values": sorted_K_max.tolist(),
            "beta_max_values": sorted_beta_max.tolist(),
            "beta_eff_values": sorted_beta_eff.tolist(),
            "beta_eff_ratio_values": sorted_beta_eff_ratio.tolist(),
            "gamma_values": sorted_gamma.tolist(),
            "scatter_delta_values": sorted_scatter_delta.tolist(),
            "tau_base_values": sorted_tau_base.tolist(),
            "tau_eff_values": sorted_tau_eff.tolist(),
            "log_tau_base_values": sorted_log_tau_base.tolist(),
            "log_tau_eff_values": sorted_log_tau_eff.tolist(),
            "condition_values": sorted_condition,
            "sub_condition_values": sorted_sub_condition,
            "split_group_values": sorted_split_group,
            "quality_values": sorted_quality,
            "phantom_flag_values": sorted_phantom_flag.tolist(),
            "metadata": sorted_metadata,
        }
    )
    save_condition_velocity_metrics_csv(
        save_condition_velocity_metrics_path,
        compute_condition_velocity_metrics(prediction_records),
    )
    save_sub_condition_velocity_metrics_csv(
        save_sub_condition_velocity_metrics_path,
        compute_sub_condition_velocity_metrics(prediction_records),
    )

    prediction_map_outputs = {}
    if SAVE_GLOBAL_PREDICTION_MAPS:
        prediction_map_outputs = save_global_prediction_maps(
            prediction_map_dir,
            test_dataset,
            sorted_metadata,
            sorted_v_pred,
            sorted_v_true,
            sorted_condition,
            sorted_sub_condition,
            spatial_shape,
            max_preview=MAX_PREDICTION_MAP_PREVIEWS,
        )

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sorted_v_true)), sorted_v_true, label="True velocity", color="black", linewidth=2)
    plt.plot(range(len(sorted_v_pred)), sorted_v_pred, label="Final d/tau velocity", color="red", linestyle="--")
    plt.plot(range(len(sorted_v_aux)), sorted_v_aux, label="Aux v_pred", color="blue", linestyle=":")
    plt.xlabel("Sample index sorted by true velocity")
    plt.ylabel("Velocity (mm/s)")
    plt.title("Clean Serial SNN-CNN Generalization")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()

    write_evaluation_report(
        report_path,
        {
            "timestamp": report_timestamp,
            "status": "completed",
            "device": str(device),
            "elapsed": time.time() - start_time,
            "model_weights_path": model_weights_path,
            "checkpoint_stage": checkpoint_stage,
            "generalization_output_dir": generalization_output_dir,
            "save_plot_path": save_plot_path,
            "save_prediction_path": save_prediction_path,
            "prediction_map_outputs": prediction_map_outputs,
            "window_ms": window_ms,
            "base_dt_us": base_dt_us,
            "base_total_steps": base_total_steps,
            "base_block_size": base_block_size,
            "snn_bin_size": snn_bin_size,
            "snn_step_us": snn_step_us,
            "snn_steps": snn_steps,
            "snn_input_scale_mode": snn_input_scale_mode,
            "batch_size": batch_size,
            "dt_us": dt_us,
            "spatial_shape": spatial_shape,
            "patch_shape": patch_shape,
            "max_velocity": max_velocity,
            "use_beta_conditioning": use_beta_conditioning,
            "use_bounded_scatter": use_bounded_scatter,
            "scatter_scale": scatter_scale,
            "max_eval_batches": max_eval_batches,
            "eval_sampling_plan": eval_sampling_plan,
            "eval_batches": len(test_loader),
            "test_ds": test_dataset,
        },
        eval_record,
    )
    print(f"=> Saved predictions to: {save_prediction_path}")
    print(f"=> Saved plot to: {save_plot_path}")
    if prediction_map_outputs:
        print(f"=> Saved mean prediction map to: {prediction_map_outputs.get('mean_prediction_map_png')}")
        print(f"=> Saved prediction map manifest to: {prediction_map_outputs.get('manifest_csv')}")
    print(f"=> Saved evaluation report to: {report_path}")


if __name__ == "__main__":
    evaluate_generalization()
