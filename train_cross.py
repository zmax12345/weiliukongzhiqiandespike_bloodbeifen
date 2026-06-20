import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


LOSS_COMPONENT_KEYS = [
    "final_velocity_loss",
    "tau_log_loss",
    "rank_loss",
    "final_var_loss",
    "v_aux_loss",
    "tau_delta_reg_loss",
]


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
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or y.size < 2:
        return float("nan")

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result

    return safe_pearson(ranks(x), ranks(y))


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


def is_better_checkpoint(val_stats, best_stats, mae_tol=0.005, metric_tol=1e-6):
    if not np.isfinite(val_stats.get("mae", float("nan"))) or not np.isfinite(val_stats.get("loss", float("nan"))):
        return False
    if best_stats is not None and val_stats.get("pred_std", 0.0) < 1e-6:
        return False
    if best_stats is None:
        return True
    if val_stats["mae"] < best_stats["mae"] - mae_tol:
        return True
    if val_stats["mae"] > best_stats["mae"] + mae_tol:
        return False
    tie_breakers = [
        ("bin_mae_max", False),
        ("max_abs_bin_bias", False),
        ("high_speed_mae", False),
        ("rank_accuracy", True),
        ("pred_std", True),
        ("loss", False),
    ]
    for key, higher_is_better in tie_breakers:
        current = val_stats.get(key, float("nan"))
        best = best_stats.get(key, float("nan"))
        if not np.isfinite(current) and not np.isfinite(best):
            continue
        if not np.isfinite(current):
            return False
        if not np.isfinite(best):
            return True
        if higher_is_better:
            if current > best + metric_tol:
                return True
            if current < best - metric_tol:
                return False
        else:
            if current < best - metric_tol:
                return True
            if current > best + metric_tol:
                return False
    return False


def is_oom_error(error):
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def unpack_batch(batch):
    if len(batch) == 6:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = batch
    else:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids = batch
        metadata = None
    return x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata


def pairwise_ranking_loss(pred, target, margin=0.12):
    target_diff = target.view(-1, 1) - target.view(1, -1)
    pred_diff = pred.view(-1, 1) - pred.view(1, -1)
    valid = target_diff > 0
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    return F.relu(margin - pred_diff[valid]).mean()


def total_variation_loss(x):
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w


def compute_training_loss(model_output, d_values, y_true, loss_weights):
    v_pred_aux = model_output.get("v_pred")
    tau_pred = model_output["tau_pred"]
    log_tau_pred = model_output["log_tau_pred"]
    log_tau_delta = model_output.get("log_tau_delta")

    tau_target = d_values / torch.clamp(y_true, min=1e-8)
    log_tau_target = torch.log(torch.clamp(tau_target, min=1e-8))
    v_final = d_values / torch.clamp(tau_pred, min=1e-8)

    loss_final_velocity = F.smooth_l1_loss(v_final, y_true)
    loss_tau_log = F.smooth_l1_loss(log_tau_pred, log_tau_target)
    loss_rank = pairwise_ranking_loss(v_final, y_true, margin=loss_weights["rank_margin"])
    final_std = v_final.std(unbiased=False)
    target_std = y_true.std(unbiased=False)
    loss_final_var = F.relu(loss_weights["pred_std_fraction"] * target_std - final_std)
    if v_pred_aux is None:
        loss_v_aux = y_true.new_tensor(0.0)
    else:
        loss_v_aux = F.smooth_l1_loss(v_pred_aux, y_true)

    if log_tau_delta is None:
        loss_tau_delta_reg = y_true.new_tensor(0.0)
    else:
        loss_tau_delta_reg = log_tau_delta.pow(2).mean()

    def weighted(key, value):
        weight = loss_weights.get(key, 0.0)
        if weight == 0.0:
            return value.new_tensor(0.0)
        return weight * value

    total_loss = (
        weighted("final_velocity", loss_final_velocity)
        + weighted("tau_log", loss_tau_log)
        + weighted("rank", loss_rank)
        + weighted("final_var", loss_final_var)
        + weighted("v_aux", loss_v_aux)
        + weighted("tau_delta_reg", loss_tau_delta_reg)
    )

    return total_loss, {
        "final_velocity_loss": loss_final_velocity,
        "tau_log_loss": loss_tau_log,
        "rank_loss": loss_rank,
        "final_var_loss": loss_final_var,
        "v_aux_loss": loss_v_aux,
        "tau_delta_reg_loss": loss_tau_delta_reg,
    }, v_final


def compute_per_velocity_stats(v_true, v_pred):
    v_true = np.asarray(v_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    rows = []
    for velocity in sorted(np.unique(v_true[np.isfinite(v_true)])):
        mask = np.isfinite(v_true) & np.isfinite(v_pred) & np.isclose(v_true, velocity, atol=1e-6)
        if not np.any(mask):
            continue
        pred_values = v_pred[mask]
        true_values = v_true[mask]
        mae, rmse, mape = compute_scalar_metrics(true_values, pred_values)
        pred_mean = float(pred_values.mean())
        rows.append(
            {
                "velocity": float(velocity),
                "samples": int(mask.sum()),
                "pred_mean": pred_mean,
                "pred_std": float(pred_values.std()),
                "bias": pred_mean - float(velocity),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )
    return rows


def summarize_per_velocity(per_velocity_rows):
    if not per_velocity_rows:
        return {
            "bin_mae_mean": float("nan"),
            "bin_mae_max": float("nan"),
            "max_abs_bin_bias": float("nan"),
            "high_speed_mae": float("nan"),
        }
    maes = np.asarray([row["mae"] for row in per_velocity_rows], dtype=np.float64)
    biases = np.asarray([row["bias"] for row in per_velocity_rows], dtype=np.float64)
    high_speed_maes = np.asarray(
        [row["mae"] for row in per_velocity_rows if row["velocity"] >= 1.5],
        dtype=np.float64,
    )
    return {
        "bin_mae_mean": float(np.nanmean(maes)),
        "bin_mae_max": float(np.nanmax(maes)),
        "max_abs_bin_bias": float(np.nanmax(np.abs(biases))),
        "high_speed_mae": float(np.nanmean(high_speed_maes)) if high_speed_maes.size else float("nan"),
    }


def compute_raw_dependency_diagnostics(v_true, v_pred, raw_total_events):
    raw = np.asarray(raw_total_events, dtype=np.float64)
    if raw.size == 0 or not np.any(np.isfinite(raw)):
        return {
            "raw_event_warning": "raw_total_events unavailable",
            "pred_vs_raw_total_events_pearson": float("nan"),
            "pred_vs_raw_total_events_spearman": float("nan"),
            "label_vs_raw_total_events_pearson": float("nan"),
            "abs_error_vs_raw_total_events_pearson": float("nan"),
            "abs_error_vs_raw_total_events_spearman": float("nan"),
        }
    v_true = np.asarray(v_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    abs_error = np.abs(v_pred - v_true)
    return {
        "raw_event_warning": "",
        "pred_vs_raw_total_events_pearson": safe_pearson(v_pred, raw),
        "pred_vs_raw_total_events_spearman": safe_spearman(v_pred, raw),
        "label_vs_raw_total_events_pearson": safe_pearson(v_true, raw),
        "abs_error_vs_raw_total_events_pearson": safe_pearson(abs_error, raw),
        "abs_error_vs_raw_total_events_spearman": safe_spearman(abs_error, raw),
    }


def save_validation_predictions_csv(path, val_stats):
    metadata_list = val_stats.get("metadata", [])
    v_true = np.asarray(val_stats.get("v_true", []), dtype=np.float64)
    v_pred = np.asarray(val_stats.get("v_pred", []), dtype=np.float64)
    v_aux = np.asarray(val_stats.get("v_aux", []), dtype=np.float64)
    d_values = np.asarray(val_stats.get("d_values", []), dtype=np.float64)
    tau_pred = np.asarray(val_stats.get("tau_pred_values", []), dtype=np.float64)
    log_tau = np.asarray(val_stats.get("log_tau_values", []), dtype=np.float64)
    rows = max(len(v_true), len(metadata_list))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "sample_id",
                "velocity_true",
                "d_value",
                "tau_pred",
                "log_tau_pred",
                "v_final",
                "v_final_clipped",
                "v_pred_aux",
                "abs_error",
                "clipped_abs_error",
                "raw_total_events",
                "normalized_total_events",
            ]
        )
        for idx in range(rows):
            meta = metadata_list[idx] if idx < len(metadata_list) and isinstance(metadata_list[idx], dict) else {}
            true_value = v_true[idx] if idx < v_true.size else float("nan")
            final_value = v_pred[idx] if idx < v_pred.size else float("nan")
            clipped_value = float(np.clip(final_value, 0.0, 2.0)) if np.isfinite(final_value) else float("nan")
            writer.writerow(
                [
                    meta.get("source_path", ""),
                    meta.get("seq_start_idx", idx),
                    true_value,
                    d_values[idx] if idx < d_values.size else float("nan"),
                    tau_pred[idx] if idx < tau_pred.size else float("nan"),
                    log_tau[idx] if idx < log_tau.size else float("nan"),
                    final_value,
                    clipped_value,
                    v_aux[idx] if idx < v_aux.size else float("nan"),
                    abs(final_value - true_value) if np.isfinite(final_value) and np.isfinite(true_value) else float("nan"),
                    abs(clipped_value - true_value) if np.isfinite(clipped_value) and np.isfinite(true_value) else float("nan"),
                    meta.get("raw_total_events", float("nan")),
                    meta.get("normalized_total_events_est", float("nan")),
                ]
            )


def format_duration(elapsed_seconds):
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"


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


def format_event_norm_source_table(title, summary):
    source_stats = summary.get("source_stats", {})
    source_scales = summary.get("source_scales", {})
    lines = [
        f"### {title}",
        "",
        "| Source | Samples | Raw Mean | Raw Std | Raw Min | Raw Max | Source Scale |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not source_stats:
        lines.append("| (empty) | 0 | 0 | 0 | 0 | 0 | 1.000000 |")
    else:
        for source, stats in source_stats.items():
            lines.append(
                f"| `{source}` | {stats.get('num_samples', 0)} | "
                f"{stats.get('mean', 0.0):.3f} | {stats.get('std', 0.0):.3f} | "
                f"{stats.get('min', 0.0):.3f} | {stats.get('max', 0.0):.3f} | "
                f"{source_scales.get(source, 1.0):.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def format_event_norm_summary_table(title, summary):
    raw = summary.get("raw_total_events", {})
    normalized = summary.get("normalized_total_events", {})
    return "\n".join(
        [
            f"### {title}",
            "",
            "| Quantity | Mean | Std | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Raw total events | {raw.get('mean', 0.0):.3f} | {raw.get('std', 0.0):.3f} | "
            f"{raw.get('min', 0.0):.3f} | {raw.get('max', 0.0):.3f} |",
            f"| Estimated normalized total events | {normalized.get('mean', 0.0):.3f} | "
            f"{normalized.get('std', 0.0):.3f} | {normalized.get('min', 0.0):.3f} | "
            f"{normalized.get('max', 0.0):.3f} |",
            "",
        ]
    )


def format_config_table(title, rows):
    lines = [f"### {title}", "", "| Parameter | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def count_trainable_parameters(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_stage_for_epoch(epoch, stage_schedule):
    cursor = 0
    for stage in stage_schedule:
        next_cursor = cursor + stage["epochs"]
        if cursor <= epoch < next_cursor:
            return stage
        cursor = next_cursor
    return stage_schedule[-1]


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
        raise ValueError(
            f"{split_name} max_batches={max_batches} is too small for source-balanced, "
            f"velocity-cycled batches across {len(sources)} source(s)."
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
        "samples_per_velocity_per_source": passes_per_source,
        "selected_samples": effective_batches * batch_size,
        "was_adjusted": max_batches is not None and effective_batches != max_batches,
    }


def build_source_velocity_loader(dataset, batch_size, num_workers, collate_fn, max_batches, split_name, epoch_idx):
    plan = compute_source_velocity_sampling_plan(dataset, batch_size, max_batches, split_name)
    sampler = SourceVelocityBatchSampler(
        dataset.source_velocity_sample_indices,
        plan,
        seed=20260427 + epoch_idx,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    ), plan


def write_training_report(report_path, run_info, epoch_records):
    train_ds = run_info["train_ds"]
    val_ds = run_info["val_ds"]
    train_sampling_plan = run_info["train_sampling_plan"]
    val_sampling_plan = run_info["val_sampling_plan"]

    lines = [
        "# Training Report",
        "",
        f"- Run timestamp: `{run_info['timestamp']}`",
        f"- Status: `{run_info['status']}`",
        f"- Device: `{run_info['device']}`",
        f"- Duration: `{format_duration(run_info['elapsed'])}`",
        f"- Best epoch: `{run_info['best_epoch']}`",
        f"- Best validation loss: `{run_info['best_val_loss']:.6f}`" if run_info["best_epoch"] >= 0 else "- Best validation loss: `N/A`",
        f"- Best validation final MAE: `{run_info['best_val_mae']:.6f}`" if run_info["best_epoch"] >= 0 else "- Best validation final MAE: `N/A`",
        "- Best checkpoint tie-breakers when `val_final_mae` is within `0.005`: `val_bin_mae_max`, `val_max_abs_bin_bias`, `val_high_speed_mae`, `final_rank_accuracy`, `final_pred_std`, `val_loss`",
        f"- Model weights path: `{run_info['model_weights_path']}`",
        f"- Best validation prediction CSV: `{run_info.get('best_val_predictions_path', '')}`",
        f"- Loss curve path: `{run_info['loss_curve_path']}`",
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
        f"- requested_batch_size: `{run_info.get('requested_batch_size', run_info['batch_size'])}`",
        f"- effective_batch_size: `{run_info['batch_size']}`",
        f"- oom_fallback_used: `{run_info.get('oom_fallback_used', False)}`",
        f"- epochs: `{run_info['epochs']}`",
        f"- dt_us: `{run_info['dt_us']}`",
        f"- spatial_shape: `{run_info['spatial_shape']}`",
        f"- patch_shape: `{run_info['patch_shape']}`",
        f"- max_train_batches: `{run_info['max_train_batches']}`",
        f"- max_val_batches: `{run_info['max_val_batches']}`",
        f"- max_velocity: `{run_info['max_velocity']}`",
        "",
        format_config_table(
            "Training Parameters",
            [
                ("optimizer", run_info["optimizer_name"]),
                ("optimizer_lr", run_info["optimizer_lr"]),
                ("scheduler", run_info["scheduler_name"]),
                ("scheduler_mode", run_info["scheduler_mode"]),
                ("scheduler_factor", run_info["scheduler_factor"]),
                ("scheduler_patience", run_info["scheduler_patience"]),
                ("gradient_clip_max_norm", run_info["gradient_clip_max_norm"]),
                ("num_workers", run_info["num_workers"]),
                ("omp_num_threads", run_info["omp_num_threads"]),
                ("trainable_total_parameters", run_info["trainable_total_parameters"]),
            ],
        ),
        format_config_table(
            "Stage Schedule",
            [
                (
                    stage["name"],
                    f"epochs={stage['epochs']}, lr={stage['lr']}, loss_weights={stage['loss_weights']}",
                )
                for stage in run_info["stage_schedule"]
            ],
        ),
        "### Batch Limit Summary",
        "",
        "| Split | Mode | Requested Max Batches | Effective Max Batches | Sources | Velocities Per Batch | Batches Per Source | Samples Per Source | Adjusted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| Train | `{train_sampling_plan['sampling_mode']}` | "
        f"{train_sampling_plan['requested_batches'] if train_sampling_plan['requested_batches'] is not None else 'all'} | "
        f"{train_sampling_plan['effective_batches']} | {train_sampling_plan['num_sources']} | "
        f"{train_sampling_plan['velocities_per_batch']} | {train_sampling_plan['batches_per_source']} | "
        f"{train_sampling_plan['samples_per_source']} | {'yes' if train_sampling_plan['was_adjusted'] else 'no'} |",
        f"| Val | `{val_sampling_plan['sampling_mode']}` | "
        f"{val_sampling_plan['requested_batches'] if val_sampling_plan['requested_batches'] is not None else 'all'} | "
        f"{val_sampling_plan['effective_batches']} | {val_sampling_plan['num_sources']} | "
        f"{val_sampling_plan['velocities_per_batch']} | {val_sampling_plan['batches_per_source']} | "
        f"{val_sampling_plan['samples_per_source']} | {'yes' if val_sampling_plan['was_adjusted'] else 'no'} |",
        "",
        "## Dataset Summary",
        "",
        f"- train_samples: `{len(train_ds)}`",
        f"- train_batches: `{run_info['train_batches']}`",
        f"- val_samples: `{len(val_ds)}`",
        f"- val_batches: `{run_info['val_batches']}`",
        "",
        "## Data Config",
        "",
        "### Train Env Config",
        "",
    ]

    for path, d_val in run_info["train_env_config"].items():
        lines.append(f"- `{path}` -> d=`{d_val}`")

    lines.extend(["", "### Val Env Config", ""])
    for path, d_val in run_info["val_env_config"].items():
        lines.append(f"- `{path}` -> d=`{d_val}`")

    lines.extend(
        [
            "",
            format_markdown_table("Train Samples Per Source", train_ds.source_sample_counts, "Source"),
            format_markdown_table("Train Samples Per Velocity", train_ds.velocity_sample_counts, "Velocity"),
            format_markdown_table("Val Samples Per Source", val_ds.source_sample_counts, "Source"),
            format_markdown_table("Val Samples Per Velocity", val_ds.velocity_sample_counts, "Velocity"),
            "## Event Intensity Normalization",
            "",
            f"- event_norm_mode: `{run_info['event_norm_mode']}`",
            f"- event_norm_clip: `{run_info['event_norm_clip']}`",
            f"- train_event_intensity_jitter_range: `{run_info['train_event_intensity_jitter_range']}`",
            "- Training uses source-level event normalization only; no sample-scale normalization is used.",
            "",
            format_event_norm_summary_table("Train Event Count Summary", train_ds.get_event_norm_summary()),
            format_event_norm_source_table("Train Source Scales", train_ds.get_event_norm_summary()),
            format_event_norm_summary_table("Val Event Count Summary", val_ds.get_event_norm_summary()),
            format_event_norm_source_table("Val Source Scales", val_ds.get_event_norm_summary()),
        ]
    )

    smoke = run_info.get("smoke_test", {})
    if smoke:
        lines.extend(
            [
                "## Smoke Test",
                "",
                f"- passed: `{smoke.get('passed')}`",
                f"- reason: `{smoke.get('reason', '')}`",
                f"- v_final_shape: `{smoke.get('v_final_shape')}`",
                f"- v_final_range: `{smoke.get('v_final_min', float('nan')):.6f}-{smoke.get('v_final_max', float('nan')):.6f}`",
                f"- v_final_std: `{smoke.get('v_final_std', float('nan')):.6e}`",
                f"- v_pred_shape: `{smoke.get('v_pred_shape')}`",
                f"- tau_pred_shape: `{smoke.get('tau_pred_shape')}`",
                f"- v_pred_range: `{smoke.get('v_pred_min', float('nan')):.6f}-{smoke.get('v_pred_max', float('nan')):.6f}`",
                f"- v_pred_std: `{smoke.get('v_pred_std', float('nan')):.6e}`",
                f"- tau_pred_range: `{smoke.get('tau_pred_min', float('nan')):.6e}-{smoke.get('tau_pred_max', float('nan')):.6e}`",
                f"- layer1_spike_rate: `{smoke.get('layer1_spike_rate', float('nan')):.6e}`",
                f"- layer2_spike_rate: `{smoke.get('layer2_spike_rate', float('nan')):.6e}`",
                f"- layer3_spike_rate: `{smoke.get('layer3_spike_rate', float('nan')):.6e}`",
                f"- feat1_std: `{smoke.get('feat1_std', float('nan')):.6e}`",
                f"- feat2_std: `{smoke.get('feat2_std', float('nan')):.6e}`",
                f"- feat3_std: `{smoke.get('feat3_std', float('nan')):.6e}`",
                f"- cnn_embedding_std: `{smoke.get('cnn_embedding_std', float('nan')):.6e}`",
                "",
            ]
        )

    epoch_columns = [
        "Epoch", "Stage", "LR", "Train Batches", "Val Batches", "Train Samples", "Val Samples",
        "Train Loss", "Train Final Velocity Loss", "Train Tau Log Loss",
        "Train Rank Loss", "Train Final Var Loss", "Train Aux V Loss",
        "Val Loss", "Val Final MAE", "Val Final RMSE", "Val Final MAPE", "Final Pred Std", "Final Pred Range",
        "Final Pred/Label Pearson", "Final Rank Acc",
        "Val Bin MAE Mean", "Val Bin MAE Max", "Val Max Abs Bin Bias", "Val High Speed MAE",
        "Aux V MAE", "Aux V Std", "Aux V Range", "Tau Sample Range", "Log Tau Pred Range",
        "Feat1 Mean", "Feat1 Std", "Feat2 Mean", "Feat2 Std", "Feat3 Mean", "Feat3 Std",
        "CNN Embedding Std", "Layer1 Spike Rate", "Layer2 Spike Rate", "Layer3 Spike Rate",
    ]
    lines.extend(
        [
            "## Epoch History",
            "",
            "| " + " | ".join(epoch_columns) + " |",
            "| " + " | ".join(["---"] * len(epoch_columns)) + " |",
        ]
    )

    if not epoch_records:
        lines.append("| " + " | ".join(["-"] * len(epoch_columns)) + " |")
    else:
        for record in epoch_records:
            lines.append(
                f"| {record['epoch']} | {record['stage']} | {record['lr']:.6e} | "
                f"{record['train_batches_processed']}/{record['train_batches_available']} | "
                f"{record['val_batches_processed']}/{record['val_batches_available']} | "
                f"{record['train_samples_seen']}/{record['train_samples_available']} | "
                f"{record['val_samples_seen']}/{record['val_samples_available']} | "
                f"{record['train_loss']:.6f} | {record['train_final_velocity_loss']:.6f} | "
                f"{record['train_tau_log_loss']:.6f} | "
                f"{record['train_rank_loss']:.6f} | {record['train_final_var_loss']:.6f} | "
                f"{record['train_v_aux_loss']:.6f} | "
                f"{record['val_loss']:.6f} | {record['val_mae']:.6f} | {record['val_rmse']:.6f} | "
                f"{record['val_mape']:.2f}% | {record['val_pred_std']:.6f} | "
                f"{record['val_pred_min']:.6f}-{record['val_pred_max']:.6f} | "
                f"{record['val_pred_label_pearson']:.6f} | {record['val_rank_accuracy']:.6f} | "
                f"{record['val_bin_mae_mean']:.6f} | {record['val_bin_mae_max']:.6f} | "
                f"{record['val_max_abs_bin_bias']:.6f} | {record['val_high_speed_mae']:.6f} | "
                f"{record['val_aux_v_mae']:.6f} | {record['val_aux_v_std']:.6f} | "
                f"{record['val_aux_v_min']:.6f}-{record['val_aux_v_max']:.6f} | "
                f"{record['val_tau_sample_min']:.6e}-{record['val_tau_sample_max']:.6e} | "
                f"{record['val_log_tau_pred_min']:.6f}-{record['val_log_tau_pred_max']:.6f} | "
                f"{record['val_feat_1_mean']:.6e} | {record['val_feat_1_std']:.6e} | "
                f"{record['val_feat_2_mean']:.6e} | {record['val_feat_2_std']:.6e} | "
                f"{record['val_feat_3_mean']:.6e} | {record['val_feat_3_std']:.6e} | "
                f"{record['val_cnn_embedding_std']:.6e} | "
                f"{record['val_layer1_spike_rate']:.6e} | {record['val_layer2_spike_rate']:.6e} | "
                f"{record['val_layer3_spike_rate']:.6e} |"
            )

    best_per_velocity = run_info.get("best_per_velocity", [])
    lines.extend(
        [
            "",
            "## Best Epoch Per-Velocity Validation",
            "",
            "| Velocity | Samples | Pred Mean | Pred Std | Bias | MAE | RMSE | MAPE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if best_per_velocity:
        for row in best_per_velocity:
            lines.append(
                f"| {row['velocity']:.6f} | {row['samples']} | {row['pred_mean']:.6f} | "
                f"{row['pred_std']:.6f} | {row['bias']:.6f} | {row['mae']:.6f} | "
                f"{row['rmse']:.6f} | {row['mape']:.2f}% |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - |")

    raw_diag = run_info.get("best_raw_dependency", {})
    lines.extend(
        [
            "",
            "## Best Epoch Raw Event Dependency",
            "",
            f"- warning: `{raw_diag.get('raw_event_warning', '')}`",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key in (
        "pred_vs_raw_total_events_pearson",
        "pred_vs_raw_total_events_spearman",
        "label_vs_raw_total_events_pearson",
        "abs_error_vs_raw_total_events_pearson",
        "abs_error_vs_raw_total_events_spearman",
    ):
        value = raw_diag.get(key, float("nan"))
        lines.append(f"| `{key}` | {value:.6f} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Final prediction is `d_values / tau_pred` from the sample-level tau head.",
            "- `v_pred` is auxiliary only and is not used for checkpoint selection.",
            f"- Training uses {run_info['window_ms']}ms window and {run_info['snn_step_us']}us pseudo-frame.",
            "- Legacy SNN neuron with kernel_norm normalization is used.",
            "- Train event intensity jitter is enabled only for train split.",
            "- No raw direct head, no teacher distillation, no fusion, no beta, no patch tau map.",
            f"- Dataset keeps {run_info['base_dt_us']}us base bins; each SNN step is a "
            f"{run_info['snn_input_scale_mode']}-scaled aggregate of {run_info['snn_bin_size']} base bins.",
            "- SNN feature diagnostics report accumulated post-SNN feature maps before CNN decoding.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _append_feature_diagnostics(output, sums, batch_count):
    for idx, key in enumerate(("snn_feat_1", "snn_feat_2", "snn_feat_3"), start=1):
        feat = output[key].detach()
        sums[f"feat_{idx}_mean"] += float(feat.mean().cpu())
        sums[f"feat_{idx}_std"] += float(feat.std(unbiased=False).cpu())
        sums[f"layer{idx}_spike_rate"] += float(output.get(f"layer{idx}_spike_rate", float("nan")))
    sums["cnn_embedding_std"] += float(output["cnn_embedding"].detach().std(unbiased=False).cpu())
    return batch_count + 1


def run_epoch(
    model,
    data_loader,
    optimizer,
    device,
    base_total_steps,
    base_block_size,
    snn_bin_size,
    snn_input_scale_mode,
    base_dt_us,
    spatial_shape,
    patch_shape,
    epoch_idx,
    split_name,
    max_batches=None,
    loss_weights=None,
    gradient_clip_max_norm=1.0,
):
    is_train = optimizer is not None
    model.train(is_train)
    loss_weights = loss_weights or {}

    epoch_loss = 0.0
    component_loss_sums = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
    feature_diag_sums = {
        "feat_1_mean": 0.0,
        "feat_1_std": 0.0,
        "feat_2_mean": 0.0,
        "feat_2_std": 0.0,
        "feat_3_mean": 0.0,
        "feat_3_std": 0.0,
        "cnn_embedding_std": 0.0,
        "layer1_spike_rate": 0.0,
        "layer2_spike_rate": 0.0,
        "layer3_spike_rate": 0.0,
    }
    feature_diag_batches = 0

    all_v_true = []
    all_v_final = []
    all_v_aux = []
    all_tau_pred = []
    all_log_tau_pred = []
    all_d_values = []
    all_metadata = []
    all_raw_total_events = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    max_batches = len(data_loader) if max_batches is None else min(max_batches, len(data_loader))
    processed_batches = 0
    progress_bar = tqdm(
        enumerate(data_loader),
        total=max_batches,
        desc=f"Epoch {epoch_idx} [{split_name}]",
        leave=False,
        dynamic_ncols=True,
    )

    with context:
        for batch_idx, batch in progress_bar:
            if batch_idx >= max_batches:
                break
            x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = unpack_batch(batch)

            y_true = y_true.to(device)
            d_values = d_values.to(device)

            manager = DenseBlockManager(
                x_seq_sparse_data,
                batch_size=y_true.shape[0],
                spatial_shape=spatial_shape,
                patch_shape=patch_shape,
            )

            if is_train:
                optimizer.zero_grad()

            model_output = model(
                dataloader_or_generator=manager,
                base_total_steps=base_total_steps,
                base_block_size=base_block_size,
                snn_bin_size=snn_bin_size,
                snn_input_scale_mode=snn_input_scale_mode,
                base_dt_us=base_dt_us,
            )
            loss, component_losses, v_final = compute_training_loss(
                model_output,
                d_values,
                y_true,
                loss_weights,
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()

            processed_batches += 1
            epoch_loss += float(loss.item())
            for key, value in component_losses.items():
                component_loss_sums[key] += float(value.item())
            feature_diag_batches = _append_feature_diagnostics(model_output, feature_diag_sums, feature_diag_batches)

            v_pred_aux = model_output["v_pred"]
            all_v_true.extend(y_true.detach().cpu().numpy().tolist())
            all_v_final.extend(v_final.detach().cpu().numpy().tolist())
            all_v_aux.extend(v_pred_aux.detach().cpu().numpy().tolist())
            all_tau_pred.extend(model_output["tau_pred"].detach().cpu().numpy().tolist())
            all_log_tau_pred.extend(model_output["log_tau_pred"].detach().cpu().numpy().tolist())
            all_d_values.extend(d_values.detach().cpu().numpy().tolist())
            if metadata is not None:
                all_metadata.extend(metadata)
                all_raw_total_events.extend(
                    [
                        float(meta.get("raw_total_events", float("nan"))) if isinstance(meta, dict) else float("nan")
                        for meta in metadata
                    ]
                )
            else:
                all_raw_total_events.extend([float("nan")] * int(y_true.shape[0]))

            v_batch = v_final.detach().cpu()
            aux_batch = v_pred_aux.detach().cpu()
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                final=f"{v_batch.min().item():.3f}-{v_batch.max().item():.3f}",
                final_std=f"{v_batch.std(unbiased=False).item():.3e}",
                aux=f"{aux_batch.min().item():.3f}-{aux_batch.max().item():.3f}",
                feat3=f"{model_output['snn_feat_3'].detach().mean().item():.2e}",
            )

    progress_bar.close()

    avg_loss = epoch_loss / max(processed_batches, 1)
    avg_component_losses = {
        key: value / max(processed_batches, 1)
        for key, value in component_loss_sums.items()
    }
    avg_feature_diag = {
        key: value / max(feature_diag_batches, 1)
        for key, value in feature_diag_sums.items()
    }

    mae, rmse, mape = compute_scalar_metrics(all_v_true, all_v_final)
    aux_mae, aux_rmse, aux_mape = compute_scalar_metrics(all_v_true, all_v_aux)
    v_true_arr = np.asarray(all_v_true, dtype=np.float64)
    v_final_arr = np.asarray(all_v_final, dtype=np.float64)
    v_aux_arr = np.asarray(all_v_aux, dtype=np.float64)
    tau_pred_arr = np.asarray(all_tau_pred, dtype=np.float64)
    log_tau_arr = np.asarray(all_log_tau_pred, dtype=np.float64)

    pred_std = float(v_final_arr.std()) if v_final_arr.size else float("nan")
    pred_min = float(v_final_arr.min()) if v_final_arr.size else float("nan")
    pred_max = float(v_final_arr.max()) if v_final_arr.size else float("nan")
    pred_label_pearson = safe_pearson(v_final_arr, v_true_arr)
    rank_accuracy = pairwise_rank_accuracy(v_final_arr, v_true_arr)
    aux_pred_std = float(v_aux_arr.std()) if v_aux_arr.size else float("nan")
    aux_pred_min = float(v_aux_arr.min()) if v_aux_arr.size else float("nan")
    aux_pred_max = float(v_aux_arr.max()) if v_aux_arr.size else float("nan")
    aux_pred_label_pearson = safe_pearson(v_aux_arr, v_true_arr)
    tau_sample_min = float(tau_pred_arr.min()) if tau_pred_arr.size else float("nan")
    tau_sample_max = float(tau_pred_arr.max()) if tau_pred_arr.size else float("nan")
    log_tau_min = float(log_tau_arr.min()) if log_tau_arr.size else float("nan")
    log_tau_max = float(log_tau_arr.max()) if log_tau_arr.size else float("nan")
    per_velocity_rows = compute_per_velocity_stats(v_true_arr, v_final_arr)
    per_velocity_summary = summarize_per_velocity(per_velocity_rows)
    raw_dependency = compute_raw_dependency_diagnostics(v_true_arr, v_final_arr, all_raw_total_events)

    return {
        "loss": avg_loss,
        **avg_component_losses,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pred_std": pred_std,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_label_pearson": pred_label_pearson,
        "rank_accuracy": rank_accuracy,
        "aux_v_mae": aux_mae,
        "aux_v_rmse": aux_rmse,
        "aux_v_mape": aux_mape,
        "aux_v_std": aux_pred_std,
        "aux_v_min": aux_pred_min,
        "aux_v_max": aux_pred_max,
        "aux_v_label_pearson": aux_pred_label_pearson,
        "tau_sample_min": tau_sample_min,
        "tau_sample_max": tau_sample_max,
        "log_tau_pred_min": log_tau_min,
        "log_tau_pred_max": log_tau_max,
        "per_velocity": per_velocity_rows,
        **per_velocity_summary,
        "raw_dependency": raw_dependency,
        **avg_feature_diag,
        "processed_batches": processed_batches,
        "available_batches": len(data_loader),
        "num_samples": len(all_v_true),
        "available_samples": len(data_loader.dataset),
        "v_true": all_v_true,
        "v_pred": all_v_final,
        "v_aux": all_v_aux,
        "d_values": all_d_values,
        "tau_pred_values": all_tau_pred,
        "log_tau_values": all_log_tau_pred,
        "metadata": all_metadata,
        "raw_total_events": all_raw_total_events,
    }


def run_smoke_test(
    model,
    data_loader,
    device,
    base_total_steps,
    base_block_size,
    snn_bin_size,
    snn_input_scale_mode,
    base_dt_us,
    spatial_shape,
    patch_shape,
):
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = unpack_batch(batch)
        y_true = y_true.to(device)
        d_values = d_values.to(device)
        manager = DenseBlockManager(
            x_seq_sparse_data,
            batch_size=y_true.shape[0],
            spatial_shape=spatial_shape,
            patch_shape=patch_shape,
        )
        output = model(
            dataloader_or_generator=manager,
            base_total_steps=base_total_steps,
            base_block_size=base_block_size,
            snn_bin_size=snn_bin_size,
            snn_input_scale_mode=snn_input_scale_mode,
            base_dt_us=base_dt_us,
        )

    v_pred = output["v_pred"].detach()
    tau_pred = output["tau_pred"].detach()
    v_final = d_values / torch.clamp(tau_pred, min=1e-8)
    feat1_std = float(output["snn_feat_1"].detach().std(unbiased=False).cpu())
    feat2_std = float(output["snn_feat_2"].detach().std(unbiased=False).cpu())
    feat3_std = float(output["snn_feat_3"].detach().std(unbiased=False).cpu())
    cnn_embedding_std = float(output["cnn_embedding"].detach().std(unbiased=False).cpu())
    stats = {
        "passed": False,
        "reason": "",
        "v_pred_shape": list(v_pred.shape),
        "v_final_shape": list(v_final.shape),
        "tau_pred_shape": list(tau_pred.shape),
        "v_final_min": float(v_final.min().cpu()),
        "v_final_max": float(v_final.max().cpu()),
        "v_final_std": float(v_final.std(unbiased=False).cpu()),
        "v_pred_min": float(v_pred.min().cpu()),
        "v_pred_max": float(v_pred.max().cpu()),
        "v_pred_std": float(v_pred.std(unbiased=False).cpu()),
        "tau_pred_min": float(tau_pred.min().cpu()),
        "tau_pred_max": float(tau_pred.max().cpu()),
        "layer1_spike_rate": float(output["layer1_spike_rate"]),
        "layer2_spike_rate": float(output["layer2_spike_rate"]),
        "layer3_spike_rate": float(output["layer3_spike_rate"]),
        "feat1_std": feat1_std,
        "feat2_std": feat2_std,
        "feat3_std": feat3_std,
        "cnn_embedding_std": cnn_embedding_std,
    }
    expected_shape = [int(y_true.shape[0])]
    checks = [
        (stats["v_final_shape"] == expected_shape, "v_final shape mismatch"),
        (stats["tau_pred_shape"] == expected_shape, "tau_pred shape mismatch"),
        (stats["layer1_spike_rate"] > 0.0, "layer1 spike rate is zero"),
        (stats["layer2_spike_rate"] > 0.0, "layer2 spike rate is zero"),
        (stats["layer3_spike_rate"] > 0.0, "layer3 spike rate is zero"),
        (stats["feat1_std"] > 0.0, "feat1 std is zero"),
        (stats["feat2_std"] > 0.0, "feat2 std is zero"),
        (stats["feat3_std"] > 0.0, "feat3 std is zero"),
        (np.isfinite(stats["cnn_embedding_std"]), "cnn embedding std is not finite"),
    ]
    failed = [message for ok, message in checks if not ok]
    if failed:
        stats["reason"] = "; ".join(failed)
    else:
        stats["passed"] = True
        stats["reason"] = "ok"
    return stats


def run_backward_oom_probe(
    model,
    data_loader,
    device,
    base_total_steps,
    base_block_size,
    snn_bin_size,
    snn_input_scale_mode,
    base_dt_us,
    spatial_shape,
    patch_shape,
    loss_weights,
):
    model.train(True)
    batch = next(iter(data_loader))
    x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = unpack_batch(batch)
    y_true = y_true.to(device)
    d_values = d_values.to(device)
    manager = DenseBlockManager(
        x_seq_sparse_data,
        batch_size=y_true.shape[0],
        spatial_shape=spatial_shape,
        patch_shape=patch_shape,
    )
    model.zero_grad(set_to_none=True)
    output = model(
        dataloader_or_generator=manager,
        base_total_steps=base_total_steps,
        base_block_size=base_block_size,
        snn_bin_size=snn_bin_size,
        snn_input_scale_mode=snn_input_scale_mode,
        base_dt_us=base_dt_us,
    )
    loss, _, _ = compute_training_loss(output, d_values, y_true, loss_weights)
    loss.backward()
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train clean Legacy-SNN -> CNN tau-final model.")
    parser.add_argument("--snn-bin-size", type=int, default=40, help="Number of 20us base bins aggregated per SNN step.")
    return parser.parse_args()


def train_cross_env(args=None):
    if args is None:
        args = parse_args()
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dt_us = 20
    window_ms = 200
    base_total_steps = int(window_ms * 1000 / base_dt_us)
    base_block_size = 400
    snn_bin_size = int(args.snn_bin_size)
    if snn_bin_size <= 0:
        raise ValueError(f"snn_bin_size must be a positive integer, got {snn_bin_size}.")
    if base_total_steps % snn_bin_size != 0:
        raise ValueError(
            f"base_total_steps ({base_total_steps}) must be divisible by snn_bin_size ({snn_bin_size})."
        )
    snn_step_us = base_dt_us * snn_bin_size
    snn_steps = base_total_steps // snn_bin_size
    snn_input_scale_mode = "sqrt"
    requested_batch_size = 4
    batch_size = requested_batch_size
    oom_fallback_used = False
    num_workers = 0
    spatial_shape = (100, 368)
    patch_shape = (50, 46)
    dt_us = base_dt_us
    max_velocity = 2.0
    max_train_batches = 60
    max_val_batches = None
    event_norm_mode = "source_scale"
    event_norm_clip = (0.25, 4.0)
    train_event_intensity_jitter_range = (0.95, 1.05)
    val_event_intensity_jitter_range = None
    optimizer_name = "Adam"
    scheduler_name = "ReduceLROnPlateau"
    scheduler_mode = "min"
    scheduler_factor = 0.5
    scheduler_patience = 5
    gradient_clip_max_norm = 1.0
    if base_total_steps != 10000:
        raise ValueError("Input config must keep 200ms window and 20us base dt.")
    if base_total_steps % base_block_size != 0:
        raise ValueError(
            f"base_total_steps ({base_total_steps}) must be divisible by base_block_size ({base_block_size})."
        )
    if base_block_size % snn_bin_size != 0:
        raise ValueError(
            f"base_block_size ({base_block_size}) must be divisible by snn_bin_size ({snn_bin_size}) "
            "so each dense block can be reshaped into SNN steps."
        )
    print(
        "SNN temporal aggregation: "
        f"snn_bin_size={snn_bin_size}, snn_step_us={snn_step_us}, "
        f"snn_steps={snn_steps}, snn_input_scale_mode={snn_input_scale_mode}"
    )

    main_loss_weights = {
        "final_velocity": 1.0,
        "tau_log": 0.75,
        "rank": 0.5,
        "final_var": 0.5,
        "v_aux": 0.1,
        "tau_delta_reg": 0.001,
        "rank_margin": 0.12,
        "pred_std_fraction": 0.8,
    }
    stage_schedule = [
        {
            "name": "stage1_warm",
            "epochs": 8,
            "lr": 1e-4,
            "loss_weights": main_loss_weights,
        },
        {
            "name": "stage2_stable",
            "epochs": 20,
            "lr": 5e-5,
            "loss_weights": main_loss_weights,
        },
        {
            "name": "stage3_finetune",
            "epochs": 12,
            "lr": 2e-5,
            "loss_weights": main_loss_weights,
        },
        {
            "name": "stage4_refine",
            "epochs": 10,
            "lr": 1e-5,
            "loss_weights": main_loss_weights,
        },
    ]
    epochs = sum(stage["epochs"] for stage in stage_schedule)
    optimizer_lr = stage_schedule[0]["lr"]

    train_env_config = {
        #"/data/zm/Moshaboli/new_data/no1": 0.018938,
        #"/data/zm/Moshaboli/new_data/no4": 0.01973,
        #"/data/zm/Moshaboli/new_data/no2": 0.01942,
        "/data/zm/2026.1.12_testdata/1.15_150_680W": 0.010419,
        "/data/zm/2026.1.12_testdata/1.15_150_580W": 0.01139,
        "/data/zm/2026.1.12_testdata/1.26_PINN_result/2.4/data": 0.00987924,
        #"/data/zm/2026.1.12_testdata/gaoyuzhi": 0.01449,
    }
    val_env_config = {
        #"/data/zm/Moshaboli/new_data/no3": 0.01963,
        "/data/zm/2026.1.12_testdata/2.3": 0.01001661,
        #"/data/zm/2026.1.12_testdata/1.15_150_580W": 0.01139,
    }

    mask_path = "/data/zm/2026.1.12_testdata/noblood/blood_maskmosha_hot_pixel_mask.npy"
    model_weights_path = "/data/zm/2026.1.12_testdata/noblood/model/best_blood_flow_model.pth"
    loss_curve_path = "/data/zm/2026.1.12_testdata/noblood/loss_curve/spike_blood_loss_curve.png"
    report_dir = "/data/zm/2026.1.12_testdata/noblood/markdown"
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"train_cross_{report_timestamp}.md")
    best_val_predictions_path = os.path.join(report_dir, "best_val_predictions.csv")

    os.makedirs(os.path.dirname(model_weights_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

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
        event_intensity_jitter_range=train_event_intensity_jitter_range,
        return_metadata=True,
    )
    train_event_norm_stats = train_ds.get_reference_event_norm_stats()
    val_ds = FlexibleBloodFlowDataset(
        val_env_config,
        mask_path=mask_path,
        T=1,
        seq_len=base_total_steps,
        dt_us=dt_us,
        max_velocity=max_velocity,
        event_norm_mode=event_norm_mode,
        event_norm_stats=train_event_norm_stats,
        event_norm_reference_mean=train_event_norm_stats["reference_mean_events_per_sample"],
        event_norm_clip=event_norm_clip,
        event_intensity_jitter_range=val_event_intensity_jitter_range,
        return_metadata=True,
    )

    def build_state_for_batch_size(candidate_batch_size, candidate_max_train_batches):
        candidate_train_plan = compute_source_velocity_sampling_plan(
            train_ds,
            candidate_batch_size,
            candidate_max_train_batches,
            "Train",
        )
        candidate_val_loader, candidate_val_plan = build_source_velocity_loader(
            dataset=val_ds,
            batch_size=candidate_batch_size,
            num_workers=num_workers,
            collate_fn=sequence_sparse_collate,
            max_batches=max_val_batches,
            split_name="Val",
            epoch_idx=0,
        )
        candidate_model = SNN_CNN_Hybrid(in_channels=1, max_velocity=max_velocity).to(device)
        candidate_optimizer = torch.optim.Adam(candidate_model.parameters(), lr=optimizer_lr)
        candidate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            candidate_optimizer,
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
        )
        candidate_smoke_loader, _ = build_source_velocity_loader(
            dataset=train_ds,
            batch_size=candidate_batch_size,
            num_workers=num_workers,
            collate_fn=sequence_sparse_collate,
            max_batches=candidate_max_train_batches,
            split_name="Train",
            epoch_idx=0,
        )
        candidate_smoke = run_smoke_test(
            candidate_model,
            candidate_smoke_loader,
            device,
            base_total_steps,
            base_block_size,
            snn_bin_size,
            snn_input_scale_mode,
            base_dt_us,
            spatial_shape,
            patch_shape,
        )
        return (
            candidate_train_plan,
            candidate_val_loader,
            candidate_val_plan,
            candidate_model,
            candidate_optimizer,
            candidate_scheduler,
            candidate_smoke,
        )

    try:
        (
            train_sampling_plan,
            val_loader,
            val_sampling_plan,
            model,
            optimizer,
            scheduler,
            smoke_test_stats,
        ) = build_state_for_batch_size(batch_size, max_train_batches)
    except RuntimeError as error:
        if not is_oom_error(error) or batch_size == 2:
            raise
        print("Batch size 4 smoke test OOM; falling back to batch size 2.")
        oom_fallback_used = True
        batch_size = 2
        max_train_batches = 120
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (
            train_sampling_plan,
            val_loader,
            val_sampling_plan,
            model,
            optimizer,
            scheduler,
            smoke_test_stats,
        ) = build_state_for_batch_size(batch_size, max_train_batches)

    print(f"Smoke test | {smoke_test_stats}")
    if not smoke_test_stats["passed"]:
        raise RuntimeError(f"Smoke test failed: {smoke_test_stats['reason']}")
    try:
        backward_probe_loader, _ = build_source_velocity_loader(
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=sequence_sparse_collate,
            max_batches=max_train_batches,
            split_name="Train",
            epoch_idx=0,
        )
        run_backward_oom_probe(
            model,
            backward_probe_loader,
            device,
            base_total_steps,
            base_block_size,
            snn_bin_size,
            snn_input_scale_mode,
            base_dt_us,
            spatial_shape,
            patch_shape,
            main_loss_weights,
        )
    except RuntimeError as error:
        if not is_oom_error(error) or batch_size == 2:
            raise
        print("Batch size 4 backward probe OOM; falling back to batch size 2.")
        oom_fallback_used = True
        batch_size = 2
        max_train_batches = 120
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (
            train_sampling_plan,
            val_loader,
            val_sampling_plan,
            model,
            optimizer,
            scheduler,
            smoke_test_stats,
        ) = build_state_for_batch_size(batch_size, max_train_batches)
        if not smoke_test_stats["passed"]:
            raise RuntimeError(f"Smoke test failed after fallback: {smoke_test_stats['reason']}")
        fallback_probe_loader, _ = build_source_velocity_loader(
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=sequence_sparse_collate,
            max_batches=max_train_batches,
            split_name="Train",
            epoch_idx=0,
        )
        run_backward_oom_probe(
            model,
            fallback_probe_loader,
            device,
            base_total_steps,
            base_block_size,
            snn_bin_size,
            snn_input_scale_mode,
            base_dt_us,
            spatial_shape,
            patch_shape,
            main_loss_weights,
        )

    trainable_total_parameters = count_trainable_parameters(model)
    print(
        f"Dataset summary | train_samples={len(train_ds)}, "
        f"train_batches={train_sampling_plan['effective_batches']}, "
        f"val_samples={len(val_ds)}, val_batches={val_sampling_plan['effective_batches']}"
    )
    print(f"Batch size | requested={requested_batch_size}, effective={batch_size}, oom_fallback={oom_fallback_used}")
    print(f"Stage schedule | {stage_schedule}")
    print(f"Training report will be saved to {report_path}")

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_checkpoint_stats = None
    best_epoch = -1
    best_per_velocity_rows = []
    best_raw_dependency = {}
    epoch_records = []
    run_status = "completed"
    active_stage_name = None

    try:
        for epoch in range(epochs):
            current_stage = get_stage_for_epoch(epoch, stage_schedule)
            if current_stage["name"] != active_stage_name:
                active_stage_name = current_stage["name"]
                set_optimizer_lr(optimizer, current_stage["lr"])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=scheduler_mode,
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                )
                print(f"\n===== Enter {current_stage['name']} | lr={current_stage['lr']} =====")

            print(f"\n===== Epoch {epoch} [{current_stage['name']}] =====")
            train_loader, train_loader_info = build_source_velocity_loader(
                dataset=train_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=sequence_sparse_collate,
                max_batches=max_train_batches,
                split_name="Train",
                epoch_idx=epoch,
            )
            print(
                f"Velocity-cycled train batches | requested="
                f"{train_loader_info['requested_batches'] if train_loader_info['requested_batches'] is not None else 'all'}, "
                f"effective={train_loader_info['effective_batches']}, "
                f"batches_per_source={train_loader_info['batches_per_source']}, "
                f"velocities_per_batch={train_loader_info['velocities_per_batch']}"
            )

            train_stats = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                base_total_steps,
                base_block_size,
                snn_bin_size,
                snn_input_scale_mode,
                base_dt_us,
                spatial_shape,
                patch_shape,
                epoch,
                "Train",
                max_batches=max_train_batches,
                loss_weights=current_stage["loss_weights"],
                gradient_clip_max_norm=gradient_clip_max_norm,
            )
            val_stats = run_epoch(
                model,
                val_loader,
                None,
                device,
                base_total_steps,
                base_block_size,
                snn_bin_size,
                snn_input_scale_mode,
                base_dt_us,
                spatial_shape,
                patch_shape,
                epoch,
                "Val",
                max_batches=max_val_batches,
                loss_weights=current_stage["loss_weights"],
                gradient_clip_max_norm=gradient_clip_max_norm,
            )

            scheduler.step(val_stats["loss"])
            if val_stats["layer1_spike_rate"] < 1e-8:
                run_status = "failed_snn_dead"
            train_loss_history.append(train_stats["loss"])
            val_loss_history.append(val_stats["loss"])

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_records.append(
                {
                    "epoch": epoch,
                    "stage": current_stage["name"],
                    "lr": current_lr,
                    "train_batches_processed": train_stats["processed_batches"],
                    "train_batches_available": train_stats["available_batches"],
                    "val_batches_processed": val_stats["processed_batches"],
                    "val_batches_available": val_stats["available_batches"],
                    "train_samples_seen": train_stats["num_samples"],
                    "train_samples_available": train_stats["available_samples"],
                    "val_samples_seen": val_stats["num_samples"],
                    "val_samples_available": val_stats["available_samples"],
                    "train_loss": train_stats["loss"],
                    "train_final_velocity_loss": train_stats["final_velocity_loss"],
                    "train_tau_log_loss": train_stats["tau_log_loss"],
                    "train_rank_loss": train_stats["rank_loss"],
                    "train_final_var_loss": train_stats["final_var_loss"],
                    "train_v_aux_loss": train_stats["v_aux_loss"],
                    "val_loss": val_stats["loss"],
                    "val_mae": val_stats["mae"],
                    "val_rmse": val_stats["rmse"],
                    "val_mape": val_stats["mape"],
                    "val_pred_std": val_stats["pred_std"],
                    "val_pred_min": val_stats["pred_min"],
                    "val_pred_max": val_stats["pred_max"],
                    "val_pred_label_pearson": val_stats["pred_label_pearson"],
                    "val_rank_accuracy": val_stats["rank_accuracy"],
                    "val_bin_mae_mean": val_stats["bin_mae_mean"],
                    "val_bin_mae_max": val_stats["bin_mae_max"],
                    "val_max_abs_bin_bias": val_stats["max_abs_bin_bias"],
                    "val_high_speed_mae": val_stats["high_speed_mae"],
                    "val_aux_v_mae": val_stats["aux_v_mae"],
                    "val_aux_v_std": val_stats["aux_v_std"],
                    "val_aux_v_min": val_stats["aux_v_min"],
                    "val_aux_v_max": val_stats["aux_v_max"],
                    "val_tau_sample_min": val_stats["tau_sample_min"],
                    "val_tau_sample_max": val_stats["tau_sample_max"],
                    "val_log_tau_pred_min": val_stats["log_tau_pred_min"],
                    "val_log_tau_pred_max": val_stats["log_tau_pred_max"],
                    "val_feat_1_mean": val_stats["feat_1_mean"],
                    "val_feat_1_std": val_stats["feat_1_std"],
                    "val_feat_2_mean": val_stats["feat_2_mean"],
                    "val_feat_2_std": val_stats["feat_2_std"],
                    "val_feat_3_mean": val_stats["feat_3_mean"],
                    "val_feat_3_std": val_stats["feat_3_std"],
                    "val_cnn_embedding_std": val_stats["cnn_embedding_std"],
                    "val_layer1_spike_rate": val_stats["layer1_spike_rate"],
                    "val_layer2_spike_rate": val_stats["layer2_spike_rate"],
                    "val_layer3_spike_rate": val_stats["layer3_spike_rate"],
                }
            )

            print(
                f"Epoch {epoch} summary | "
                f"train_loss={train_stats['loss']:.6f}, val_loss={val_stats['loss']:.6f}, "
                f"final_mae={val_stats['mae']:.6f}, final_rmse={val_stats['rmse']:.6f}, "
                f"final_mape={val_stats['mape']:.2f}%, final_std={val_stats['pred_std']:.6f}, "
                f"final_range=[{val_stats['pred_min']:.6f}, {val_stats['pred_max']:.6f}], "
                f"final_corr={val_stats['pred_label_pearson']:.6f}, final_rank={val_stats['rank_accuracy']:.6f}, "
                f"bin_mae_max={val_stats['bin_mae_max']:.6f}, max_bin_bias={val_stats['max_abs_bin_bias']:.6f}, "
                f"high_speed_mae={val_stats['high_speed_mae']:.6f}, "
                f"aux_mae={val_stats['aux_v_mae']:.6f}, aux_std={val_stats['aux_v_std']:.6f}, "
                f"aux_range=[{val_stats['aux_v_min']:.6f}, {val_stats['aux_v_max']:.6f}], "
                f"log_tau=[{val_stats['log_tau_pred_min']:.6f}, {val_stats['log_tau_pred_max']:.6f}], "
                f"spike_rates=[{val_stats['layer1_spike_rate']:.3e}, {val_stats['layer2_spike_rate']:.3e}, "
                f"{val_stats['layer3_spike_rate']:.3e}], "
                f"feat1_std={val_stats['feat_1_std']:.3e}, feat2_std={val_stats['feat_2_std']:.3e}, "
                f"feat3_std={val_stats['feat_3_std']:.3e}, emb_std={val_stats['cnn_embedding_std']:.3e}"
            )

            if is_better_checkpoint(val_stats, best_checkpoint_stats):
                best_val_loss = val_stats["loss"]
                best_val_mae = val_stats["mae"]
                best_checkpoint_stats = {
                    "mae": val_stats["mae"],
                    "loss": val_stats["loss"],
                    "pred_std": val_stats["pred_std"],
                    "rank_accuracy": val_stats["rank_accuracy"],
                    "bin_mae_max": val_stats["bin_mae_max"],
                    "max_abs_bin_bias": val_stats["max_abs_bin_bias"],
                    "high_speed_mae": val_stats["high_speed_mae"],
                }
                best_per_velocity_rows = val_stats["per_velocity"]
                best_raw_dependency = val_stats["raw_dependency"]
                best_epoch = epoch
                save_validation_predictions_csv(best_val_predictions_path, val_stats)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "stage": current_stage["name"],
                        "epoch": epoch,
                        "val_loss": val_stats["loss"],
                        "val_mae": val_stats["mae"],
                        "val_rmse": val_stats["rmse"],
                        "val_pred_std": val_stats["pred_std"],
                        "val_pred_min": val_stats["pred_min"],
                        "val_pred_max": val_stats["pred_max"],
                        "val_pred_label_pearson": val_stats["pred_label_pearson"],
                        "val_rank_accuracy": val_stats["rank_accuracy"],
                        "val_bin_mae_mean": val_stats["bin_mae_mean"],
                        "val_bin_mae_max": val_stats["bin_mae_max"],
                        "val_max_abs_bin_bias": val_stats["max_abs_bin_bias"],
                        "val_high_speed_mae": val_stats["high_speed_mae"],
                        "val_per_velocity": val_stats["per_velocity"],
                        "val_raw_dependency": val_stats["raw_dependency"],
                        "val_aux_v_mae": val_stats["aux_v_mae"],
                        "val_aux_v_std": val_stats["aux_v_std"],
                        "val_aux_v_min": val_stats["aux_v_min"],
                        "val_aux_v_max": val_stats["aux_v_max"],
                        "val_tau_sample_min": val_stats["tau_sample_min"],
                        "val_tau_sample_max": val_stats["tau_sample_max"],
                        "val_log_tau_pred_min": val_stats["log_tau_pred_min"],
                        "val_log_tau_pred_max": val_stats["log_tau_pred_max"],
                        "stage_schedule": stage_schedule,
                        "input_config": {
                            "window_ms": window_ms,
                            "base_dt_us": base_dt_us,
                            "base_total_steps": base_total_steps,
                            "base_block_size": base_block_size,
                            "snn_bin_size": snn_bin_size,
                            "snn_step_us": snn_step_us,
                            "snn_steps": snn_steps,
                            "snn_input_scale_mode": snn_input_scale_mode,
                        },
                        "event_norm_stats": train_event_norm_stats,
                        "event_norm_config": {
                            "event_norm_mode": event_norm_mode,
                            "event_norm_clip": event_norm_clip,
                            "train_event_intensity_jitter_range": train_event_intensity_jitter_range,
                            "max_velocity": max_velocity,
                        },
                        "max_velocity": max_velocity,
                    },
                    model_weights_path,
                )
                print(f"Saved new best model to {model_weights_path}")

            run_info = {
                "timestamp": report_timestamp,
                "status": run_status,
                "device": str(device),
                "elapsed": time.time() - start_time,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
                "model_weights_path": model_weights_path,
                "loss_curve_path": loss_curve_path,
                "best_val_predictions_path": best_val_predictions_path,
                "window_ms": window_ms,
                "base_dt_us": base_dt_us,
                "base_total_steps": base_total_steps,
                "base_block_size": base_block_size,
                "snn_bin_size": snn_bin_size,
                "snn_step_us": snn_step_us,
                "snn_steps": snn_steps,
                "snn_input_scale_mode": snn_input_scale_mode,
                "requested_batch_size": requested_batch_size,
                "batch_size": batch_size,
                "oom_fallback_used": oom_fallback_used,
                "epochs": epochs,
                "dt_us": dt_us,
                "num_workers": num_workers,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                "spatial_shape": spatial_shape,
                "patch_shape": patch_shape,
                "max_velocity": max_velocity,
                "max_train_batches": max_train_batches,
                "max_val_batches": max_val_batches,
                "event_norm_mode": event_norm_mode,
                "event_norm_clip": event_norm_clip,
                "train_event_intensity_jitter_range": train_event_intensity_jitter_range,
                "optimizer_name": optimizer_name,
                "optimizer_lr": optimizer_lr,
                "scheduler_name": scheduler_name,
                "scheduler_mode": scheduler_mode,
                "scheduler_factor": scheduler_factor,
                "scheduler_patience": scheduler_patience,
                "gradient_clip_max_norm": gradient_clip_max_norm,
                "trainable_total_parameters": trainable_total_parameters,
                "stage_schedule": stage_schedule,
                "train_sampling_plan": train_sampling_plan,
                "val_sampling_plan": val_sampling_plan,
                "train_batches": train_sampling_plan["effective_batches"],
                "val_batches": val_sampling_plan["effective_batches"],
                "train_env_config": train_env_config,
                "val_env_config": val_env_config,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "smoke_test": smoke_test_stats,
                "best_per_velocity": best_per_velocity_rows,
                "best_raw_dependency": best_raw_dependency,
            }
            write_training_report(report_path, run_info, epoch_records)
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("Training interrupted by user.")

    if train_loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_loss_history)), train_loss_history, label="Train Loss", color="blue", marker="o")
        plt.plot(range(len(val_loss_history)), val_loss_history, label="Validation Loss", color="red", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Clean Serial SNN-CNN Training Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()

    elapsed = time.time() - start_time
    run_info = {
        "timestamp": report_timestamp,
        "status": run_status,
        "device": str(device),
        "elapsed": elapsed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "model_weights_path": model_weights_path,
        "loss_curve_path": loss_curve_path,
        "best_val_predictions_path": best_val_predictions_path,
        "window_ms": window_ms,
        "base_dt_us": base_dt_us,
        "base_total_steps": base_total_steps,
        "base_block_size": base_block_size,
        "snn_bin_size": snn_bin_size,
        "snn_step_us": snn_step_us,
        "snn_steps": snn_steps,
        "snn_input_scale_mode": snn_input_scale_mode,
        "requested_batch_size": requested_batch_size,
        "batch_size": batch_size,
        "oom_fallback_used": oom_fallback_used,
        "epochs": epochs,
        "dt_us": dt_us,
        "num_workers": num_workers,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "spatial_shape": spatial_shape,
        "patch_shape": patch_shape,
        "max_velocity": max_velocity,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "event_norm_mode": event_norm_mode,
        "event_norm_clip": event_norm_clip,
        "train_event_intensity_jitter_range": train_event_intensity_jitter_range,
        "optimizer_name": optimizer_name,
        "optimizer_lr": optimizer_lr,
        "scheduler_name": scheduler_name,
        "scheduler_mode": scheduler_mode,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "gradient_clip_max_norm": gradient_clip_max_norm,
        "trainable_total_parameters": trainable_total_parameters,
        "stage_schedule": stage_schedule,
        "train_sampling_plan": train_sampling_plan,
        "val_sampling_plan": val_sampling_plan,
        "train_batches": train_sampling_plan["effective_batches"],
        "val_batches": val_sampling_plan["effective_batches"],
        "train_env_config": train_env_config,
        "val_env_config": val_env_config,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "smoke_test": smoke_test_stats,
        "best_per_velocity": best_per_velocity_rows,
        "best_raw_dependency": best_raw_dependency,
    }
    write_training_report(report_path, run_info, epoch_records)

    print("\n" + "=" * 60)
    print(f"Training finished in {format_duration(elapsed)}")
    print(f"Best epoch: {best_epoch}")
    if best_epoch >= 0:
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best validation MAE: {best_val_mae:.6f}")
    else:
        print("Best validation loss: N/A")
        print("Best validation MAE: N/A")
    print(f"Saved loss curve to: {loss_curve_path}")
    print(f"Saved training report to: {report_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_cross_env(parse_args())
