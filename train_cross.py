import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import math
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

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


LOSS_COMPONENT_KEYS = [
    "final_velocity_loss",
    "tau_log_loss",
    "tau_eff_loss",
    "tau_base_anchor_loss",
    "rank_loss",
    "final_var_loss",
    "v_aux_loss",
    "tau_delta_reg_loss",
    "scatter_delta_reg_loss",
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
    elif len(batch) == 6:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = batch
        batch_size = int(y_true.shape[0])
        K_max = torch.ones(batch_size, dtype=torch.float32)
        beta_max = torch.ones(batch_size, dtype=torch.float32)
        log_beta_max = torch.zeros(batch_size, dtype=torch.float32)
        condition = ["unknown"] * batch_size
        sub_condition = ["unknown"] * batch_size
        split_group = ["train_val"] * batch_size
        quality = ["legacy"] * batch_size
        phantom_flag = torch.zeros(batch_size, dtype=torch.float32)
    else:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids = batch
        metadata = None
        batch_size = int(y_true.shape[0])
        K_max = torch.ones(batch_size, dtype=torch.float32)
        beta_max = torch.ones(batch_size, dtype=torch.float32)
        log_beta_max = torch.zeros(batch_size, dtype=torch.float32)
        condition = ["unknown"] * batch_size
        sub_condition = ["unknown"] * batch_size
        split_group = ["train_val"] * batch_size
        quality = ["legacy"] * batch_size
        phantom_flag = torch.zeros(batch_size, dtype=torch.float32)
    return (
        x_seq_sparse_data,
        y_true,
        d_values,
        env_maps,
        source_ids,
        metadata,
        K_max,
        beta_max,
        log_beta_max,
        condition,
        sub_condition,
        split_group,
        quality,
        phantom_flag,
    )


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
    log_tau_base = model_output.get("log_tau_base", log_tau_pred)
    scatter_delta = model_output.get("scatter_delta")

    tau_target = d_values / torch.clamp(y_true, min=1e-8)
    log_tau_target = torch.log(torch.clamp(tau_target, min=1e-8))
    v_final = d_values / torch.clamp(tau_pred, min=1e-8)

    loss_final_velocity = F.smooth_l1_loss(v_final, y_true)
    loss_tau_log = F.smooth_l1_loss(log_tau_pred, log_tau_target)
    loss_tau_base_anchor = F.smooth_l1_loss(log_tau_base, log_tau_target)
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
    if scatter_delta is None:
        loss_scatter_delta_reg = y_true.new_tensor(0.0)
    else:
        loss_scatter_delta_reg = scatter_delta.pow(2).mean()

    def weighted(key, value):
        weight = loss_weights.get(key, 0.0)
        if weight == 0.0:
            return value.new_tensor(0.0)
        return weight * value

    total_loss = (
        weighted("final_velocity", loss_final_velocity)
        + weighted("tau_log", loss_tau_log)
        + weighted("tau_eff", loss_tau_log)
        + weighted("tau_base_anchor", loss_tau_base_anchor)
        + weighted("rank", loss_rank)
        + weighted("final_var", loss_final_var)
        + weighted("v_aux", loss_v_aux)
        + weighted("tau_delta_reg", loss_tau_delta_reg)
        + weighted("scatter_delta_reg", loss_scatter_delta_reg)
    )

    return total_loss, {
        "final_velocity_loss": loss_final_velocity,
        "tau_log_loss": loss_tau_log,
        "tau_eff_loss": loss_tau_log,
        "tau_base_anchor_loss": loss_tau_base_anchor,
        "rank_loss": loss_rank,
        "final_var_loss": loss_final_var,
        "v_aux_loss": loss_v_aux,
        "tau_delta_reg_loss": loss_tau_delta_reg,
        "scatter_delta_reg_loss": loss_scatter_delta_reg,
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


def summarize_array(prefix, values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }


def compute_per_condition_stats(conditions, phantom_flags, v_true, v_pred, gamma, beta_eff_ratio, scatter_delta):
    v_true = np.asarray(v_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta_eff_ratio = np.asarray(beta_eff_ratio, dtype=np.float64)
    scatter_delta = np.asarray(scatter_delta, dtype=np.float64)
    labels = []
    for idx in range(len(v_true)):
        cond = conditions[idx] if idx < len(conditions) else "unknown"
        flag = phantom_flags[idx] if idx < len(phantom_flags) else float("nan")
        labels.append(f"{cond}|phantom={flag:g}" if np.isfinite(flag) else str(cond))
    rows = []
    for label in sorted(set(labels)):
        mask = np.asarray([item == label for item in labels], dtype=bool)
        finite = mask & np.isfinite(v_true) & np.isfinite(v_pred)
        if not np.any(finite):
            continue
        err = v_pred[finite] - v_true[finite]
        rows.append(
            {
                "condition": label,
                "samples": int(finite.sum()),
                "mae": float(np.abs(err).mean()),
                "bias": float(err.mean()),
                "pred_mean": float(v_pred[finite].mean()),
                "label_mean": float(v_true[finite].mean()),
                "gamma_mean": float(np.nanmean(gamma[finite])) if gamma.size == v_true.size else float("nan"),
                "beta_eff_ratio_mean": float(np.nanmean(beta_eff_ratio[finite])) if beta_eff_ratio.size == v_true.size else float("nan"),
                "scatter_delta_mean": float(np.nanmean(scatter_delta[finite])) if scatter_delta.size == v_true.size else float("nan"),
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


def safe_float(value, default=float("nan")):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def list_value(values, idx, default=float("nan")):
    if values is None or idx >= len(values):
        return default
    return values[idx]


def finite_mean(values):
    arr = np.asarray([safe_float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_std(values):
    arr = np.asarray([safe_float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def infer_condition(condition, phantom_flag):
    if condition is not None and str(condition).strip() and str(condition).lower() != "nan":
        return str(condition)
    flag = safe_float(phantom_flag)
    if np.isfinite(flag):
        if int(round(flag)) == 0:
            return "nof"
        if int(round(flag)) == 1:
            return "withf"
    return "unknown"


def build_prediction_records(val_stats):
    metadata_list = val_stats.get("metadata", [])
    v_true = val_stats.get("v_true", [])
    v_pred = val_stats.get("v_pred", [])
    v_aux = val_stats.get("v_aux", [])
    d_values = val_stats.get("d_values", [])
    tau_pred = val_stats.get("tau_pred_values", [])
    log_tau = val_stats.get("log_tau_values", [])
    K_max = val_stats.get("K_max_values", [])
    beta_max = val_stats.get("beta_max_values", [])
    gamma = val_stats.get("gamma_values", [])
    beta_eff = val_stats.get("beta_eff_values", [])
    beta_eff_ratio = val_stats.get("beta_eff_ratio_values", [])
    scatter_delta = val_stats.get("scatter_delta_values", [])
    tau_base = val_stats.get("tau_base_values", [])
    tau_eff = val_stats.get("tau_eff_values", [])
    log_tau_base = val_stats.get("log_tau_base_values", [])
    log_tau_eff = val_stats.get("log_tau_eff_values", [])
    conditions = val_stats.get("condition_values", [])
    sub_conditions = val_stats.get("sub_condition_values", [])
    split_groups = val_stats.get("split_group_values", [])
    qualities = val_stats.get("quality_values", [])
    phantom_flags = val_stats.get("phantom_flag_values", [])
    source_ids = val_stats.get("source_id_values", [])
    raw_total_events_values = val_stats.get("raw_total_events", [])
    split = str(val_stats.get("split_name", "val")).lower()
    epoch = val_stats.get("epoch_idx", float("nan"))

    rows = max(
        len(v_true),
        len(v_pred),
        len(metadata_list),
        len(d_values),
        len(tau_pred),
        len(conditions),
        len(source_ids),
    )
    records = []
    for idx in range(rows):
        meta = metadata_list[idx] if idx < len(metadata_list) and isinstance(metadata_list[idx], dict) else {}
        true_value = safe_float(list_value(v_true, idx))
        final_value = safe_float(list_value(v_pred, idx))
        d_value = safe_float(list_value(d_values, idx, meta.get("d_value", float("nan"))))
        k_value = safe_float(list_value(K_max, idx, meta.get("K_max", float("nan"))))
        beta_value = safe_float(list_value(beta_max, idx, meta.get("beta_max", float("nan"))))
        beta_eff_value = safe_float(list_value(beta_eff, idx))
        beta_ratio_value = safe_float(list_value(beta_eff_ratio, idx))
        if not np.isfinite(beta_ratio_value) and np.isfinite(beta_eff_value) and np.isfinite(beta_value):
            beta_ratio_value = beta_eff_value / (beta_value + 1e-8)
        flag_value = safe_float(list_value(phantom_flags, idx, meta.get("phantom_flag", float("nan"))), default=-1.0)
        condition_value = infer_condition(
            list_value(conditions, idx, meta.get("condition", "")),
            flag_value,
        )
        sub_condition_value = str(list_value(sub_conditions, idx, meta.get("sub_condition", "unknown")))
        split_group_value = str(list_value(split_groups, idx, meta.get("split_group", "train_val")))
        quality_value = str(list_value(qualities, idx, meta.get("quality", "")))
        raw_events = safe_float(meta.get("raw_total_events", list_value(raw_total_events_values, idx)))
        norm_events = safe_float(
            meta.get(
                "normalized_total_events",
                meta.get("normalized_total_events_est", float("nan")),
            )
        )
        error = final_value - true_value if np.isfinite(final_value) and np.isfinite(true_value) else float("nan")
        source_id_value = safe_float(list_value(source_ids, idx, meta.get("source_id", float("nan"))))
        records.append(
            {
                "split": split,
                "epoch": epoch,
                "source": meta.get("source_path", ""),
                "source_id": source_id_value,
                "sample_id": meta.get("seq_start_idx", idx),
                "file_path": meta.get("file_path", ""),
                "condition": condition_value,
                "sub_condition": sub_condition_value,
                "split_group": split_group_value,
                "quality": quality_value,
                "phantom_flag": flag_value,
                "velocity": true_value,
                "y_true": true_value,
                "pred_final": final_value,
                "v_final": final_value,
                "error": error,
                "abs_error": abs(error) if np.isfinite(error) else float("nan"),
                "d_value": d_value,
                "K_max": k_value,
                "beta_max": beta_value,
                "gamma": safe_float(list_value(gamma, idx)),
                "beta_eff": beta_eff_value,
                "beta_eff_ratio": beta_ratio_value,
                "scatter_delta": safe_float(list_value(scatter_delta, idx)),
                "tau_base": safe_float(list_value(tau_base, idx)),
                "tau_eff": safe_float(list_value(tau_eff, idx, list_value(tau_pred, idx))),
                "log_tau_base": safe_float(list_value(log_tau_base, idx)),
                "log_tau_eff": safe_float(list_value(log_tau_eff, idx, list_value(log_tau, idx))),
                "tau_pred": safe_float(list_value(tau_pred, idx)),
                "log_tau_pred": safe_float(list_value(log_tau, idx)),
                "v_pred_aux": safe_float(list_value(v_aux, idx)),
                "raw_total_events": raw_events,
                "norm_total_events": norm_events,
                "normalized_total_events": norm_events,
            }
        )
    return records


def compute_condition_velocity_metrics(records):
    groups = {}
    for record in records:
        velocity = round(safe_float(record.get("velocity", record.get("y_true"))), 6)
        if not np.isfinite(velocity):
            continue
        flag = safe_float(record.get("phantom_flag"), default=-1.0)
        condition = infer_condition(record.get("condition"), flag)
        key = (condition, int(round(flag)) if np.isfinite(flag) else -1, velocity)
        groups.setdefault(key, []).append(record)

    rows = []
    for condition, phantom_flag, velocity in sorted(groups.keys(), key=lambda item: (item[0], item[1], item[2])):
        items = groups[(condition, phantom_flag, velocity)]
        labels = np.asarray([safe_float(item.get("y_true", item.get("velocity"))) for item in items], dtype=np.float64)
        preds = np.asarray([safe_float(item.get("pred_final", item.get("v_final"))) for item in items], dtype=np.float64)
        finite_pair = np.isfinite(labels) & np.isfinite(preds)
        errors = preds[finite_pair] - labels[finite_pair]
        if errors.size:
            abs_errors = np.abs(errors)
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mape_values = abs_errors / np.maximum(np.abs(labels[finite_pair]), 1e-8) * 100.0
        else:
            abs_errors = np.asarray([], dtype=np.float64)
            rmse = float("nan")
            mape_values = np.asarray([], dtype=np.float64)
        rows.append(
            {
                "condition": condition,
                "phantom_flag": phantom_flag,
                "velocity": float(velocity),
                "samples": int(len(items)),
                "label_mean": finite_mean(labels),
                "pred_mean": finite_mean(preds),
                "pred_std": finite_std(preds),
                "bias": finite_mean(errors),
                "mae": finite_mean(abs_errors),
                "rmse": rmse,
                "mape": finite_mean(mape_values),
                "gamma_mean": finite_mean([item.get("gamma") for item in items]),
                "gamma_std": finite_std([item.get("gamma") for item in items]),
                "beta_eff_ratio_mean": finite_mean([item.get("beta_eff_ratio") for item in items]),
                "beta_eff_ratio_std": finite_std([item.get("beta_eff_ratio") for item in items]),
                "scatter_delta_mean": finite_mean([item.get("scatter_delta") for item in items]),
                "scatter_delta_std": finite_std([item.get("scatter_delta") for item in items]),
                "tau_base_mean": finite_mean([item.get("tau_base") for item in items]),
                "tau_base_std": finite_std([item.get("tau_base") for item in items]),
                "tau_eff_mean": finite_mean([item.get("tau_eff", item.get("tau_pred")) for item in items]),
                "tau_eff_std": finite_std([item.get("tau_eff", item.get("tau_pred")) for item in items]),
                "K_max": finite_mean([item.get("K_max") for item in items]),
                "beta_max": finite_mean([item.get("beta_max") for item in items]),
                "d_value": finite_mean([item.get("d_value") for item in items]),
                "raw_total_events_mean": finite_mean([item.get("raw_total_events") for item in items]),
                "raw_total_events_std": finite_std([item.get("raw_total_events") for item in items]),
                "norm_total_events_mean": finite_mean([item.get("norm_total_events") for item in items]),
                "norm_total_events_std": finite_std([item.get("norm_total_events") for item in items]),
            }
        )
    return rows


def compute_sub_condition_velocity_metrics(records):
    groups = {}
    for record in records:
        velocity = round(safe_float(record.get("velocity", record.get("y_true"))), 6)
        if not np.isfinite(velocity):
            continue
        sub_condition = str(record.get("sub_condition") or "unknown")
        condition = infer_condition(record.get("condition"), record.get("phantom_flag"))
        flag = safe_float(record.get("phantom_flag"), default=-1.0)
        key = (sub_condition, condition, int(round(flag)) if np.isfinite(flag) else -1, velocity)
        groups.setdefault(key, []).append(record)

    rows = []
    for sub_condition, condition, phantom_flag, velocity in sorted(groups.keys(), key=lambda item: (item[0], item[1], item[3])):
        items = groups[(sub_condition, condition, phantom_flag, velocity)]
        labels = np.asarray([safe_float(item.get("y_true", item.get("velocity"))) for item in items], dtype=np.float64)
        preds = np.asarray([safe_float(item.get("pred_final", item.get("v_final"))) for item in items], dtype=np.float64)
        finite_pair = np.isfinite(labels) & np.isfinite(preds)
        errors = preds[finite_pair] - labels[finite_pair]
        if errors.size:
            abs_errors = np.abs(errors)
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            mape_values = abs_errors / np.maximum(np.abs(labels[finite_pair]), 1e-8) * 100.0
        else:
            abs_errors = np.asarray([], dtype=np.float64)
            rmse = float("nan")
            mape_values = np.asarray([], dtype=np.float64)
        rows.append(
            {
                "sub_condition": sub_condition,
                "condition": condition,
                "phantom_flag": phantom_flag,
                "velocity": float(velocity),
                "samples": int(len(items)),
                "label_mean": finite_mean(labels),
                "pred_mean": finite_mean(preds),
                "pred_std": finite_std(preds),
                "bias": finite_mean(errors),
                "mae": finite_mean(abs_errors),
                "rmse": rmse,
                "mape": finite_mean(mape_values),
                "gamma_mean": finite_mean([item.get("gamma") for item in items]),
                "gamma_std": finite_std([item.get("gamma") for item in items]),
                "beta_eff_ratio_mean": finite_mean([item.get("beta_eff_ratio") for item in items]),
                "beta_eff_ratio_std": finite_std([item.get("beta_eff_ratio") for item in items]),
                "scatter_delta_mean": finite_mean([item.get("scatter_delta") for item in items]),
                "scatter_delta_std": finite_std([item.get("scatter_delta") for item in items]),
                "tau_base_mean": finite_mean([item.get("tau_base") for item in items]),
                "tau_eff_mean": finite_mean([item.get("tau_eff", item.get("tau_pred")) for item in items]),
                "K_max": finite_mean([item.get("K_max") for item in items]),
                "beta_max": finite_mean([item.get("beta_max") for item in items]),
                "d_value": finite_mean([item.get("d_value") for item in items]),
                "raw_total_events_mean": finite_mean([item.get("raw_total_events") for item in items]),
                "norm_total_events_mean": finite_mean([item.get("norm_total_events") for item in items]),
            }
        )
    return rows


def compute_per_sub_condition_stats(records):
    groups = {}
    for record in records:
        sub_condition = str(record.get("sub_condition") or "unknown")
        groups.setdefault(sub_condition, []).append(record)
    rows = []
    for sub_condition in sorted(groups):
        items = groups[sub_condition]
        labels = np.asarray([safe_float(item.get("y_true", item.get("velocity"))) for item in items], dtype=np.float64)
        preds = np.asarray([safe_float(item.get("pred_final", item.get("v_final"))) for item in items], dtype=np.float64)
        finite_pair = np.isfinite(labels) & np.isfinite(preds)
        errors = preds[finite_pair] - labels[finite_pair]
        condition = infer_condition(items[0].get("condition"), items[0].get("phantom_flag")) if items else "unknown"
        rows.append(
            {
                "sub_condition": sub_condition,
                "condition": condition,
                "samples": len(items),
                "mae": finite_mean(np.abs(errors)),
                "bias": finite_mean(errors),
                "pred_mean": finite_mean(preds),
                "label_mean": finite_mean(labels),
                "gamma_mean": finite_mean([item.get("gamma") for item in items]),
                "beta_eff_ratio_mean": finite_mean([item.get("beta_eff_ratio") for item in items]),
                "scatter_delta_mean": finite_mean([item.get("scatter_delta") for item in items]),
            }
        )
    return rows


def compute_beta_scatter_correlation_by_condition(records):
    groups = {}
    for record in records:
        flag = safe_float(record.get("phantom_flag"), default=-1.0)
        condition = infer_condition(record.get("condition"), flag)
        key = (condition, int(round(flag)) if np.isfinite(flag) else -1)
        groups.setdefault(key, []).append(record)

    rows = []
    for condition, phantom_flag in sorted(groups.keys(), key=lambda item: (item[0], item[1])):
        items = groups[(condition, phantom_flag)]
        velocities = [item.get("velocity", item.get("y_true")) for item in items]
        rows.append(
            {
                "condition": condition,
                "phantom_flag": phantom_flag,
                "samples": len(items),
                "corr_gamma_velocity": safe_pearson([item.get("gamma") for item in items], velocities),
                "corr_scatter_delta_velocity": safe_pearson([item.get("scatter_delta") for item in items], velocities),
                "corr_beta_eff_ratio_velocity": safe_pearson([item.get("beta_eff_ratio") for item in items], velocities),
            }
        )
    return rows


def save_condition_velocity_metrics_csv(path, rows):
    fieldnames = [
        "condition",
        "phantom_flag",
        "velocity",
        "samples",
        "label_mean",
        "pred_mean",
        "pred_std",
        "bias",
        "mae",
        "rmse",
        "mape",
        "gamma_mean",
        "gamma_std",
        "beta_eff_ratio_mean",
        "beta_eff_ratio_std",
        "scatter_delta_mean",
        "scatter_delta_std",
        "tau_base_mean",
        "tau_base_std",
        "tau_eff_mean",
        "tau_eff_std",
        "K_max",
        "beta_max",
        "d_value",
        "raw_total_events_mean",
        "raw_total_events_std",
        "norm_total_events_mean",
        "norm_total_events_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, float("nan")) for key in fieldnames})


def save_sub_condition_velocity_metrics_csv(path, rows):
    fieldnames = [
        "sub_condition",
        "condition",
        "phantom_flag",
        "velocity",
        "samples",
        "label_mean",
        "pred_mean",
        "pred_std",
        "bias",
        "mae",
        "rmse",
        "mape",
        "gamma_mean",
        "gamma_std",
        "beta_eff_ratio_mean",
        "beta_eff_ratio_std",
        "scatter_delta_mean",
        "scatter_delta_std",
        "tau_base_mean",
        "tau_eff_mean",
        "K_max",
        "beta_max",
        "d_value",
        "raw_total_events_mean",
        "norm_total_events_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, float("nan")) for key in fieldnames})


def save_validation_predictions_csv(path, val_stats):
    prediction_records = val_stats.get("prediction_records")
    if prediction_records is None:
        prediction_records = build_prediction_records(val_stats)
    fieldnames = [
        "split",
        "epoch",
        "source",
        "source_id",
        "sample_id",
        "file_path",
        "condition",
        "sub_condition",
        "split_group",
        "quality",
        "phantom_flag",
        "velocity",
        "y_true",
        "pred_final",
        "v_final",
        "error",
        "abs_error",
        "d_value",
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
        "tau_pred",
        "log_tau_pred",
        "v_pred_aux",
        "raw_total_events",
        "norm_total_events",
        "normalized_total_events",
        "velocity_true",
        "v_final_clipped",
        "clipped_abs_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in prediction_records:
            row = {key: record.get(key, float("nan")) for key in fieldnames}
            row["velocity_true"] = record.get("y_true", record.get("velocity", float("nan")))
            final_value = safe_float(record.get("pred_final", record.get("v_final")))
            true_value = safe_float(row["velocity_true"])
            clipped_value = float(np.clip(final_value, 0.0, 2.0)) if np.isfinite(final_value) else float("nan")
            row["v_final_clipped"] = clipped_value
            row["clipped_abs_error"] = (
                abs(clipped_value - true_value)
                if np.isfinite(clipped_value) and np.isfinite(true_value)
                else float("nan")
            )
            writer.writerow(row)


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


def collect_channel_mask_summary_rows(*datasets):
    rows_by_source = {}
    for dataset in datasets:
        if dataset is None:
            continue
        summary = getattr(dataset, "get_channel_mask_summary", lambda: {})()
        for source, row in summary.items():
            if source not in rows_by_source:
                rows_by_source[source] = {
                    "source": source,
                    "sub_condition": row.get("sub_condition", "unknown"),
                    "channel_mask_enabled": bool(row.get("channel_mask_enabled", False)),
                    "channel_mask_path": row.get("channel_mask_path", ""),
                    "channel_mask_area_pixels": row.get("channel_mask_area_pixels", float("nan")),
                    "channel_mask_area_ratio": row.get("channel_mask_area_ratio", float("nan")),
                    "events_before_channel_mask": 0,
                    "events_after_channel_mask": 0,
                }
            rows_by_source[source]["events_before_channel_mask"] += int(row.get("events_before_channel_mask", 0))
            rows_by_source[source]["events_after_channel_mask"] += int(row.get("events_after_channel_mask", 0))

    rows = []
    for row in rows_by_source.values():
        before = int(row["events_before_channel_mask"])
        after = int(row["events_after_channel_mask"])
        if row.get("channel_mask_enabled"):
            retained_ratio = float(after / max(before, 1))
        else:
            retained_ratio = 1.0
        row["channel_mask_retained_ratio"] = retained_ratio
        rows.append(row)
    return sorted(rows, key=lambda item: str(item.get("source", "")))


def format_channel_mask_summary_table(rows):
    lines = [
        "## Channel Mask Summary",
        "",
        "| Source | Sub Condition | Channel Mask Enabled | Channel Mask Path | Mask Area Pixels | Mask Area Ratio | Events Before Channel Mask | Events After Channel Mask | Retained Ratio |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| - | - | False | - | NaN | NaN | 0 | 0 | 1.000000 |")
    else:
        for row in rows:
            lines.append(
                f"| `{row.get('source', '')}` | `{row.get('sub_condition', '')}` | "
                f"`{bool(row.get('channel_mask_enabled', False))}` | `{row.get('channel_mask_path', '')}` | "
                f"{safe_float(row.get('channel_mask_area_pixels')):.0f} | "
                f"{safe_float(row.get('channel_mask_area_ratio')):.6f} | "
                f"{int(row.get('events_before_channel_mask', 0))} | "
                f"{int(row.get('events_after_channel_mask', 0))} | "
                f"{safe_float(row.get('channel_mask_retained_ratio'), 1.0):.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


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
    evaluate_ds = run_info.get("evaluate_ds")
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
        f"- Best condition-velocity metrics CSV: `{run_info.get('best_val_condition_velocity_metrics_path', '')}`",
        f"- Best sub-condition-velocity metrics CSV: `{run_info.get('best_val_sub_condition_velocity_metrics_path', '')}`",
        f"- Evaluate prediction CSV: `{run_info.get('evaluate_predictions_path', '')}`",
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
        f"- mask_path: `{run_info.get('mask_path')}`",
        f"- mask_enabled: `{run_info.get('mask_enabled')}`",
        f"- data_split_mode: `{run_info.get('data_split_mode', 'config')}`",
        f"- enable_split_group: `{run_info.get('enable_split_group', False)}`",
        f"- train_val_split_group: `{run_info.get('train_val_split_group', '')}`",
        f"- evaluate_split_group: `{run_info.get('evaluate_split_group', '')}`",
        f"- exclude_split_group: `{run_info.get('exclude_split_group', '')}`",
        f"- train_velocities: `{run_info.get('train_include_velocities')}`",
        f"- val_velocities: `{run_info.get('val_include_velocities')}`",
        f"- use_beta_conditioning: `{run_info.get('use_beta_conditioning', False)}`",
        f"- use_bounded_scatter: `{run_info.get('use_bounded_scatter', False)}`",
        f"- scatter_scale: `{run_info.get('scatter_scale', float('nan'))}`",
        f"- tau_base_anchor_weight: `{run_info.get('tau_base_anchor_weight', float('nan'))}`",
        f"- scatter_delta_reg_weight: `{run_info.get('scatter_delta_reg_weight', float('nan'))}`",
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
        f"- evaluate_samples: `{len(evaluate_ds) if evaluate_ds is not None else 0}`",
        f"- evaluate_batches: `{math.ceil(len(evaluate_ds) / max(1, run_info['batch_size'])) if evaluate_ds is not None else 0}`",
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

    split_rows = run_info.get("data_source_split_rows", [])
    lines.extend(
        [
            "",
            "## Data Source Split Summary",
            "",
            "| Source | Condition | Sub Condition | Phantom Flag | Split Group | Quality | K Max | Beta Max | D Value | Used In Train | Used In Val | Used In Evaluate |",
            "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    if split_rows:
        for row in split_rows:
            lines.append(
                f"| `{row.get('source', '')}` | `{row.get('condition', '')}` | `{row.get('sub_condition', '')}` | "
                f"{safe_float(row.get('phantom_flag'), -1):.0f} | `{row.get('split_group', '')}` | `{row.get('quality', '')}` | "
                f"{safe_float(row.get('K_max')):.6f} | {safe_float(row.get('beta_max')):.6f} | "
                f"{safe_float(row.get('d_value')):.6f} | `{row.get('used_in_train')}` | `{row.get('used_in_val')}` | "
                f"`{row.get('used_in_evaluate')}` |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - |")

    excluded_rows = [row for row in split_rows if str(row.get("split_group")) == str(run_info.get("exclude_split_group", "excluded"))]
    lines.extend(
        [
            "",
            "## Excluded Data Sources",
            "",
            "| Source | Sub Condition | Reason |",
            "| --- | --- | --- |",
        ]
    )
    if excluded_rows:
        for row in excluded_rows:
            lines.append(f"| `{row.get('source', '')}` | `{row.get('sub_condition', '')}` | `{row.get('quality', '')}` |")
    else:
        lines.append("| - | - | - |")

    lines.extend(
        [
            "",
            "## Train/Val Split",
            "",
            f"- train_velocities: `{run_info.get('train_include_velocities')}`",
            f"- val_velocities: `{run_info.get('val_include_velocities')}`",
            "",
            format_markdown_table("Train Samples Per Source", train_ds.source_sample_counts, "Source"),
            format_markdown_table("Train Samples Per Sub Condition", summarize_samples_by_metadata(train_ds, "sub_condition"), "Sub Condition"),
            format_markdown_table("Train Samples Per Velocity", train_ds.velocity_sample_counts, "Velocity"),
            format_markdown_table("Val Samples Per Source", val_ds.source_sample_counts, "Source"),
            format_markdown_table("Val Samples Per Sub Condition", summarize_samples_by_metadata(val_ds, "sub_condition"), "Sub Condition"),
            format_markdown_table("Val Samples Per Velocity", val_ds.velocity_sample_counts, "Velocity"),
            format_channel_mask_summary_table(
                run_info.get(
                    "channel_mask_summary_rows",
                    collect_channel_mask_summary_rows(train_ds, val_ds, evaluate_ds),
                )
            ),
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
                f"- beta_eff_lt_beta_max: `{smoke.get('beta_eff_lt_beta_max')}`",
                f"- gamma_range: `{smoke.get('gamma_min', float('nan')):.6f}-{smoke.get('gamma_max', float('nan')):.6f}`",
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

    best_condition_velocity = run_info.get("best_condition_velocity", [])
    lines.extend(
        [
            "",
            "## Best Epoch Per-Condition-Per-Velocity Validation",
            "",
            f"- Best condition-velocity metrics CSV: `{run_info.get('best_val_condition_velocity_metrics_path', '')}`",
            "",
            "| Condition | Phantom Flag | Velocity | Samples | Label Mean | Pred Mean | Pred Std | Bias | MAE | RMSE | MAPE | Gamma Mean | Gamma Std | Beta Eff Ratio Mean | Beta Eff Ratio Std | Scatter Delta Mean | Scatter Delta Std | Tau Base Mean | Tau Base Std | Tau Eff Mean | Tau Eff Std | K Max | Beta Max | D Value | Raw Total Events Mean | Raw Total Events Std | Norm Total Events Mean | Norm Total Events Std |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if best_condition_velocity:
        for row in best_condition_velocity:
            lines.append(
                f"| `{row.get('condition', 'unknown')}` | {row.get('phantom_flag', -1)} | "
                f"{row.get('velocity', float('nan')):.6f} | {row.get('samples', 0)} | "
                f"{row.get('label_mean', float('nan')):.6f} | {row.get('pred_mean', float('nan')):.6f} | "
                f"{row.get('pred_std', float('nan')):.6f} | {row.get('bias', float('nan')):.6f} | "
                f"{row.get('mae', float('nan')):.6f} | {row.get('rmse', float('nan')):.6f} | "
                f"{row.get('mape', float('nan')):.2f}% | {row.get('gamma_mean', float('nan')):.6f} | "
                f"{row.get('gamma_std', float('nan')):.6f} | {row.get('beta_eff_ratio_mean', float('nan')):.6f} | "
                f"{row.get('beta_eff_ratio_std', float('nan')):.6f} | {row.get('scatter_delta_mean', float('nan')):.6f} | "
                f"{row.get('scatter_delta_std', float('nan')):.6f} | {row.get('tau_base_mean', float('nan')):.6e} | "
                f"{row.get('tau_base_std', float('nan')):.6e} | {row.get('tau_eff_mean', float('nan')):.6e} | "
                f"{row.get('tau_eff_std', float('nan')):.6e} | {row.get('K_max', float('nan')):.6f} | "
                f"{row.get('beta_max', float('nan')):.6f} | {row.get('d_value', float('nan')):.6f} | "
                f"{row.get('raw_total_events_mean', float('nan')):.3f} | {row.get('raw_total_events_std', float('nan')):.3f} | "
                f"{row.get('norm_total_events_mean', float('nan')):.6f} | {row.get('norm_total_events_std', float('nan')):.6f} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")

    best_sub_condition_velocity = run_info.get("best_sub_condition_velocity", [])
    lines.extend(
        [
            "",
            "## Best Epoch Per-Sub-Condition-Per-Velocity Validation",
            "",
            f"- Best sub-condition-velocity metrics CSV: `{run_info.get('best_val_sub_condition_velocity_metrics_path', '')}`",
            "",
            "| Sub Condition | Condition | Phantom Flag | Velocity | Samples | Label Mean | Pred Mean | Pred Std | Bias | MAE | RMSE | MAPE | Gamma Mean | Gamma Std | Beta Eff Ratio Mean | Beta Eff Ratio Std | Scatter Delta Mean | Scatter Delta Std | Tau Base Mean | Tau Eff Mean | K Max | Beta Max | D Value | Raw Total Events Mean | Norm Total Events Mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if best_sub_condition_velocity:
        for row in best_sub_condition_velocity:
            lines.append(
                f"| `{row.get('sub_condition', 'unknown')}` | `{row.get('condition', 'unknown')}` | "
                f"{row.get('phantom_flag', -1)} | {row.get('velocity', float('nan')):.6f} | {row.get('samples', 0)} | "
                f"{row.get('label_mean', float('nan')):.6f} | {row.get('pred_mean', float('nan')):.6f} | "
                f"{row.get('pred_std', float('nan')):.6f} | {row.get('bias', float('nan')):.6f} | "
                f"{row.get('mae', float('nan')):.6f} | {row.get('rmse', float('nan')):.6f} | "
                f"{row.get('mape', float('nan')):.2f}% | {row.get('gamma_mean', float('nan')):.6f} | "
                f"{row.get('gamma_std', float('nan')):.6f} | {row.get('beta_eff_ratio_mean', float('nan')):.6f} | "
                f"{row.get('beta_eff_ratio_std', float('nan')):.6f} | {row.get('scatter_delta_mean', float('nan')):.6f} | "
                f"{row.get('scatter_delta_std', float('nan')):.6f} | {row.get('tau_base_mean', float('nan')):.6e} | "
                f"{row.get('tau_eff_mean', float('nan')):.6e} | {row.get('K_max', float('nan')):.6f} | "
                f"{row.get('beta_max', float('nan')):.6f} | {row.get('d_value', float('nan')):.6f} | "
                f"{row.get('raw_total_events_mean', float('nan')):.3f} | {row.get('norm_total_events_mean', float('nan')):.6f} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")

    best_corr_by_condition = run_info.get("best_beta_scatter_correlation_by_condition", [])
    lines.extend(
        [
            "",
            "## Best Epoch Beta/Scatter Correlation By Condition",
            "",
            "| Condition | Phantom Flag | Samples | Corr Gamma Velocity | Corr Scatter Delta Velocity | Corr Beta Eff Ratio Velocity |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if best_corr_by_condition:
        for row in best_corr_by_condition:
            lines.append(
                f"| `{row.get('condition', 'unknown')}` | {row.get('phantom_flag', -1)} | {row.get('samples', 0)} | "
                f"{row.get('corr_gamma_velocity', float('nan')):.6f} | "
                f"{row.get('corr_scatter_delta_velocity', float('nan')):.6f} | "
                f"{row.get('corr_beta_eff_ratio_velocity', float('nan')):.6f} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")

    best_beta_diag = run_info.get("best_beta_diagnostics", {})
    lines.extend(
        [
            "",
            "## Best Epoch Beta/Scatter Diagnostics",
            "",
            f"- beta conditioning enabled: `{run_info.get('use_beta_conditioning', False)}`",
            f"- bounded scatter enabled: `{run_info.get('use_bounded_scatter', False)}`",
            f"- scatter_scale: `{run_info.get('scatter_scale', float('nan'))}`",
            f"- beta_eff_lt_beta_max: `{best_beta_diag.get('beta_eff_lt_beta_max', 'N/A')}`",
            f"- gamma_mean/std/range: `{best_beta_diag.get('gamma_mean', float('nan')):.6f}` / `{best_beta_diag.get('gamma_std', float('nan')):.6f}` / `{best_beta_diag.get('gamma_min', float('nan')):.6f}-{best_beta_diag.get('gamma_max', float('nan')):.6f}`",
            f"- beta_eff_ratio_mean/std: `{best_beta_diag.get('beta_eff_ratio_mean', float('nan')):.6f}` / `{best_beta_diag.get('beta_eff_ratio_std', float('nan')):.6f}`",
            f"- scatter_delta_mean/std/range: `{best_beta_diag.get('scatter_delta_mean', float('nan')):.6f}` / `{best_beta_diag.get('scatter_delta_std', float('nan')):.6f}` / `{best_beta_diag.get('scatter_delta_min', float('nan')):.6f}-{best_beta_diag.get('scatter_delta_max', float('nan')):.6f}`",
            f"- tau_base_mean/range: `{best_beta_diag.get('tau_base_mean', float('nan')):.6e}` / `{best_beta_diag.get('tau_base_min', float('nan')):.6e}-{best_beta_diag.get('tau_base_max', float('nan')):.6e}`",
            f"- tau_eff_mean/range: `{best_beta_diag.get('tau_eff_mean', float('nan')):.6e}` / `{best_beta_diag.get('tau_eff_min', float('nan')):.6e}-{best_beta_diag.get('tau_eff_max', float('nan')):.6e}`",
            f"- corr_gamma_velocity: `{best_beta_diag.get('corr_gamma_velocity', float('nan')):.6f}`",
            f"- corr_scatter_delta_velocity: `{best_beta_diag.get('corr_scatter_delta_velocity', float('nan')):.6f}`",
            "",
            "### Best Epoch Per-Condition Validation",
            "",
            "| Condition | Samples | MAE | Bias | Pred Mean | Label Mean | Gamma Mean | Beta Eff Ratio Mean | Scatter Delta Mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    best_per_condition = run_info.get("best_per_condition", [])
    if best_per_condition:
        for row in best_per_condition:
            lines.append(
                f"| `{row['condition']}` | {row['samples']} | {row['mae']:.6f} | {row['bias']:.6f} | "
                f"{row['pred_mean']:.6f} | {row['label_mean']:.6f} | {row['gamma_mean']:.6f} | "
                f"{row['beta_eff_ratio_mean']:.6f} | {row['scatter_delta_mean']:.6f} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "### Best Epoch Per-Sub-Condition Validation",
            "",
            "| Sub Condition | Condition | Samples | MAE | Bias | Pred Mean | Label Mean | Gamma Mean | Beta Eff Ratio Mean | Scatter Delta Mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    best_per_sub_condition = run_info.get("best_per_sub_condition", [])
    if best_per_sub_condition:
        for row in best_per_sub_condition:
            lines.append(
                f"| `{row.get('sub_condition', 'unknown')}` | `{row.get('condition', 'unknown')}` | {row.get('samples', 0)} | "
                f"{row.get('mae', float('nan')):.6f} | {row.get('bias', float('nan')):.6f} | "
                f"{row.get('pred_mean', float('nan')):.6f} | {row.get('label_mean', float('nan')):.6f} | "
                f"{row.get('gamma_mean', float('nan')):.6f} | {row.get('beta_eff_ratio_mean', float('nan')):.6f} | "
                f"{row.get('scatter_delta_mean', float('nan')):.6f} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")

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

    eval_stats = run_info.get("evaluate_stats")
    lines.extend(["", "## Independent Evaluate Summary", ""])
    if eval_stats:
        lines.extend(
            [
                f"- evaluate_predictions_csv: `{run_info.get('evaluate_predictions_path', '')}`",
                f"- evaluate_condition_velocity_metrics_csv: `{run_info.get('evaluate_condition_velocity_metrics_path', '')}`",
                f"- evaluate_sub_condition_velocity_metrics_csv: `{run_info.get('evaluate_sub_condition_velocity_metrics_path', '')}`",
                f"- evaluate_samples: `{eval_stats.get('num_samples', 0)}`",
                f"- evaluate_batches: `{eval_stats.get('processed_batches', 0)}`",
                f"- evaluate_final_MAE: `{eval_stats.get('mae', float('nan')):.6f}`",
                f"- evaluate_final_RMSE: `{eval_stats.get('rmse', float('nan')):.6f}`",
                f"- evaluate_final_MAPE: `{eval_stats.get('mape', float('nan')):.2f}%`",
                f"- evaluate_pred_std: `{eval_stats.get('pred_std', float('nan')):.6f}`",
                f"- evaluate_pred_range: `{eval_stats.get('pred_min', float('nan')):.6f}-{eval_stats.get('pred_max', float('nan')):.6f}`",
                f"- evaluate_rank_acc: `{eval_stats.get('rank_accuracy', float('nan')):.6f}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- evaluate was not run in this training session.",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Final prediction is `d_values / tau_pred`; `tau_pred` now denotes `tau_eff` when bounded scatter is enabled.",
            "- `v_pred` is auxiliary only and is not used for checkpoint selection.",
            f"- Training uses {run_info['window_ms']}ms window and {run_info['snn_step_us']}us pseudo-frame.",
            "- Legacy SNN neuron with kernel_norm normalization is used.",
            "- Train event intensity jitter is enabled only for train split.",
            "- K_max enters only through beta_max/log_beta_max conditioning; d_value is still supplied by data_config and is not redesigned here.",
            "- beta_eff is constrained as sigmoid(raw_gamma) * beta_max; no same-condition gamma/scatter smoothing loss is used.",
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
    all_K_max = []
    all_beta_max = []
    all_beta_eff = []
    all_beta_eff_ratio = []
    all_gamma = []
    all_scatter_delta = []
    all_tau_base = []
    all_tau_eff = []
    all_log_tau_base = []
    all_log_tau_eff = []
    all_conditions = []
    all_sub_conditions = []
    all_split_groups = []
    all_qualities = []
    all_phantom_flags = []
    all_source_ids = []

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
            (
                x_seq_sparse_data,
                y_true,
                d_values,
                env_maps,
                source_ids,
                metadata,
                K_max,
                beta_max,
                log_beta_max,
                condition,
                sub_condition,
                split_group,
                quality,
                phantom_flag,
            ) = unpack_batch(batch)

            y_true = y_true.to(device)
            d_values = d_values.to(device)
            beta_max = beta_max.to(device)
            log_beta_max = log_beta_max.to(device)

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
                beta_max=beta_max,
                log_beta_max=log_beta_max,
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
            all_K_max.extend(K_max.detach().cpu().numpy().tolist())
            all_beta_max.extend(model_output["beta_max"].detach().cpu().numpy().tolist())
            all_beta_eff.extend(model_output["beta_eff"].detach().cpu().numpy().tolist())
            all_beta_eff_ratio.extend(model_output["beta_eff_ratio"].detach().cpu().numpy().tolist())
            all_gamma.extend(model_output["gamma"].detach().cpu().numpy().tolist())
            all_scatter_delta.extend(model_output["scatter_delta"].detach().cpu().numpy().tolist())
            all_tau_base.extend(model_output["tau_base"].detach().cpu().numpy().tolist())
            all_tau_eff.extend(model_output["tau_eff"].detach().cpu().numpy().tolist())
            all_log_tau_base.extend(model_output["log_tau_base"].detach().cpu().numpy().tolist())
            all_log_tau_eff.extend(model_output["log_tau_eff"].detach().cpu().numpy().tolist())
            all_conditions.extend(condition)
            all_sub_conditions.extend(sub_condition)
            all_split_groups.extend(split_group)
            all_qualities.extend(quality)
            all_phantom_flags.extend(phantom_flag.detach().cpu().numpy().tolist())
            if hasattr(source_ids, "detach"):
                all_source_ids.extend(source_ids.detach().cpu().numpy().tolist())
            else:
                all_source_ids.extend(list(source_ids))
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
    gamma_arr = np.asarray(all_gamma, dtype=np.float64)
    beta_max_arr = np.asarray(all_beta_max, dtype=np.float64)
    beta_eff_arr = np.asarray(all_beta_eff, dtype=np.float64)
    beta_eff_ratio_arr = np.asarray(all_beta_eff_ratio, dtype=np.float64)
    scatter_delta_arr = np.asarray(all_scatter_delta, dtype=np.float64)
    tau_base_arr = np.asarray(all_tau_base, dtype=np.float64)
    tau_eff_arr = np.asarray(all_tau_eff, dtype=np.float64)
    phantom_flag_arr = np.asarray(all_phantom_flags, dtype=np.float64)

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
    beta_diag = {}
    beta_diag.update(summarize_array("gamma", gamma_arr))
    beta_diag.update(summarize_array("beta_max", beta_max_arr))
    beta_diag.update(summarize_array("beta_eff", beta_eff_arr))
    beta_diag.update(summarize_array("beta_eff_ratio", beta_eff_ratio_arr))
    beta_diag.update(summarize_array("scatter_delta", scatter_delta_arr))
    beta_diag.update(summarize_array("tau_base", tau_base_arr))
    beta_diag.update(summarize_array("tau_eff", tau_eff_arr))
    beta_diag["corr_gamma_velocity"] = safe_pearson(gamma_arr, v_true_arr)
    beta_diag["corr_scatter_delta_velocity"] = safe_pearson(scatter_delta_arr, v_true_arr)
    beta_diag["beta_eff_lt_beta_max"] = bool(
        beta_eff_arr.size > 0 and np.all(beta_eff_arr < beta_max_arr + 1e-12)
    )
    per_condition_rows = compute_per_condition_stats(
        all_conditions,
        phantom_flag_arr,
        v_true_arr,
        v_final_arr,
        gamma_arr,
        beta_eff_ratio_arr,
        scatter_delta_arr,
    )
    prediction_base_stats = {
        "epoch_idx": epoch_idx,
        "split_name": split_name,
        "v_true": all_v_true,
        "v_pred": all_v_final,
        "v_aux": all_v_aux,
        "d_values": all_d_values,
        "tau_pred_values": all_tau_pred,
        "log_tau_values": all_log_tau_pred,
        "K_max_values": all_K_max,
        "beta_max_values": all_beta_max,
        "beta_eff_values": all_beta_eff,
        "beta_eff_ratio_values": all_beta_eff_ratio,
        "gamma_values": all_gamma,
        "scatter_delta_values": all_scatter_delta,
        "tau_base_values": all_tau_base,
        "tau_eff_values": all_tau_eff,
        "log_tau_base_values": all_log_tau_base,
        "log_tau_eff_values": all_log_tau_eff,
        "condition_values": all_conditions,
        "sub_condition_values": all_sub_conditions,
        "split_group_values": all_split_groups,
        "quality_values": all_qualities,
        "phantom_flag_values": all_phantom_flags,
        "source_id_values": all_source_ids,
        "metadata": all_metadata,
    }
    prediction_records = build_prediction_records(prediction_base_stats)
    condition_velocity_rows = compute_condition_velocity_metrics(prediction_records)
    sub_condition_velocity_rows = compute_sub_condition_velocity_metrics(prediction_records)
    per_sub_condition_rows = compute_per_sub_condition_stats(prediction_records)
    beta_scatter_correlation_by_condition = compute_beta_scatter_correlation_by_condition(prediction_records)

    return {
        "epoch_idx": epoch_idx,
        "split_name": split_name,
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
        **beta_diag,
        "per_condition": per_condition_rows,
        "per_sub_condition": per_sub_condition_rows,
        "condition_velocity": condition_velocity_rows,
        "sub_condition_velocity": sub_condition_velocity_rows,
        "beta_scatter_correlation_by_condition": beta_scatter_correlation_by_condition,
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
        "K_max_values": all_K_max,
        "beta_max_values": all_beta_max,
        "beta_eff_values": all_beta_eff,
        "beta_eff_ratio_values": all_beta_eff_ratio,
        "gamma_values": all_gamma,
        "scatter_delta_values": all_scatter_delta,
        "tau_base_values": all_tau_base,
        "tau_eff_values": all_tau_eff,
        "log_tau_base_values": all_log_tau_base,
        "log_tau_eff_values": all_log_tau_eff,
        "condition_values": all_conditions,
        "sub_condition_values": all_sub_conditions,
        "split_group_values": all_split_groups,
        "quality_values": all_qualities,
        "phantom_flag_values": all_phantom_flags,
        "source_id_values": all_source_ids,
        "metadata": all_metadata,
        "raw_total_events": all_raw_total_events,
        "prediction_records": prediction_records,
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
        (
            x_seq_sparse_data,
            y_true,
            d_values,
            env_maps,
            source_ids,
            metadata,
            K_max,
            beta_max,
            log_beta_max,
            condition,
            sub_condition,
            split_group,
            quality,
            phantom_flag,
        ) = unpack_batch(batch)
        y_true = y_true.to(device)
        d_values = d_values.to(device)
        beta_max = beta_max.to(device)
        log_beta_max = log_beta_max.to(device)
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
            beta_max=beta_max,
            log_beta_max=log_beta_max,
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
        "beta_eff_lt_beta_max": bool(torch.all(output["beta_eff"] < output["beta_max"] + 1e-12).detach().cpu()),
        "gamma_min": float(output["gamma"].detach().min().cpu()),
        "gamma_max": float(output["gamma"].detach().max().cpu()),
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
        (stats["tau_pred_min"] > 0.0, "tau_pred is not strictly positive"),
        (stats["beta_eff_lt_beta_max"], "beta_eff is not bounded by beta_max"),
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
    (
        x_seq_sparse_data,
        y_true,
        d_values,
        env_maps,
        source_ids,
        metadata,
        K_max,
        beta_max,
        log_beta_max,
        condition,
        sub_condition,
        split_group,
        quality,
        phantom_flag,
    ) = unpack_batch(batch)
    y_true = y_true.to(device)
    d_values = d_values.to(device)
    beta_max = beta_max.to(device)
    log_beta_max = log_beta_max.to(device)
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
        beta_max=beta_max,
        log_beta_max=log_beta_max,
    )
    loss, _, _ = compute_training_loss(output, d_values, y_true, loss_weights)
    loss.backward()
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train clean Legacy-SNN -> CNN tau-final model.")
    parser.add_argument("--snn-bin-size", type=int, default=40, help="Number of 20us base bins aggregated per SNN step.")
    parser.add_argument("--data-split-mode", choices=["config", "velocity"], default="velocity")
    parser.add_argument("--enable_split_group", type=str2bool, default=True)
    parser.add_argument("--train_val_split_group", type=str, default="train_val")
    parser.add_argument("--evaluate_split_group", type=str, default="evaluate")
    parser.add_argument("--exclude_split_group", type=str, default="excluded")
    parser.add_argument("--run_evaluate_after_train", type=str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=None, help="Default is 4 for config mode and 2 for velocity mode.")
    parser.add_argument("--mask-path", default="/data/zm/Weiliukong/6.17/mask/blood_maskweiliukong_hot_pixel_mask.npy")
    parser.add_argument("--disable-mask", action="store_true", help="Disable hot-pixel mask filtering even if mask-path exists.")
    parser.add_argument("--train-velocities", nargs="+", type=float, default=[0.2, 0.5, 1.0, 1.2, 1.8, 2.0])
    parser.add_argument("--val-velocities", nargs="+", type=float, default=[0.8, 1.5])
    parser.add_argument("--use_beta_conditioning", type=str2bool, default=True)
    parser.add_argument("--use_bounded_scatter", type=str2bool, default=True)
    parser.add_argument("--scatter_scale", type=float, default=0.3)
    parser.add_argument("--tau_base_anchor_weight", type=float, default=0.2)
    parser.add_argument("--scatter_delta_reg_weight", type=float, default=0.02)
    return parser.parse_args()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def source_config_value(config_value, key, default=None):
    if isinstance(config_value, dict):
        return config_value.get(key, default)
    if key == "d_value":
        return config_value
    return default


def split_config_by_group(data_config, train_val_group="train_val", evaluate_group="evaluate", excluded_group="excluded", enabled=True):
    if not enabled:
        return dict(data_config), {}, {}
    train_val_config = {}
    evaluate_config = {}
    excluded_config = {}
    for path, cfg in data_config.items():
        split_group = str(source_config_value(cfg, "split_group", train_val_group))
        if split_group == train_val_group:
            train_val_config[path] = cfg
        elif split_group == evaluate_group:
            evaluate_config[path] = cfg
        elif split_group == excluded_group:
            excluded_config[path] = cfg
    return train_val_config, evaluate_config, excluded_config


def build_data_source_split_rows(data_config, train_val_group="train_val", evaluate_group="evaluate", excluded_group="excluded"):
    rows = []
    for path, cfg in data_config.items():
        k_max = safe_float(source_config_value(cfg, "K_max", source_config_value(cfg, "k_max", 1.0)))
        d_value = safe_float(source_config_value(cfg, "d_value"))
        split_group = str(source_config_value(cfg, "split_group", train_val_group))
        rows.append(
            {
                "source": path,
                "condition": str(source_config_value(cfg, "condition", "unknown")),
                "sub_condition": str(source_config_value(cfg, "sub_condition", "unknown")),
                "phantom_flag": source_config_value(cfg, "phantom_flag", -1),
                "split_group": split_group,
                "quality": str(source_config_value(cfg, "quality", "legacy")),
                "K_max": k_max,
                "beta_max": k_max * k_max if np.isfinite(k_max) else float("nan"),
                "d_value": d_value,
                "used_in_train": split_group == train_val_group,
                "used_in_val": split_group == train_val_group,
                "used_in_evaluate": split_group == evaluate_group,
            }
        )
    return rows


def summarize_samples_by_metadata(dataset, key):
    counts = {}
    for meta in getattr(dataset, "sample_metadata", []):
        value = str(meta.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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
    data_split_mode = args.data_split_mode
    requested_batch_size = int(args.batch_size) if args.batch_size is not None else (2 if data_split_mode == "velocity" else 4)
    batch_size = requested_batch_size
    oom_fallback_used = False
    num_workers = 0
    spatial_shape = (100, 1200)
    patch_shape = (50, 50)
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
    use_beta_conditioning = bool(args.use_beta_conditioning)
    use_bounded_scatter = bool(args.use_bounded_scatter)
    scatter_scale = float(args.scatter_scale)
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
        "tau_eff": 0.0,
        "tau_base_anchor": float(args.tau_base_anchor_weight),
        "scatter_delta_reg": float(args.scatter_delta_reg_weight),
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

    legacy_train_env_config = {
        #"/data/zm/Moshaboli/new_data/no1": 0.018938,
        #"/data/zm/Moshaboli/new_data/no4": 0.01973,
        #"/data/zm/Moshaboli/new_data/no2": 0.01942,
        "/data/zm/2026.1.12_testdata/1.15_150_680W": {
            "d_value": 0.010419,
            "K_max": 1.0,
            "condition": "nof",
            "phantom_flag": 0,
        },
        "/data/zm/2026.1.12_testdata/1.15_150_580W": {
            "d_value": 0.01139,
            "K_max": 1.0,
            "condition": "nof",
            "phantom_flag": 0,
        },
        "/data/zm/2026.1.12_testdata/1.26_PINN_result/2.4/data": {
            "d_value": 0.00987924,
            "K_max": 1.0,
            "condition": "nof",
            "phantom_flag": 0,
        },
        #"/data/zm/2026.1.12_testdata/gaoyuzhi": 0.01449,
    }
    legacy_val_env_config = {
        #"/data/zm/Moshaboli/new_data/no3": 0.01963,
        "/data/zm/2026.1.12_testdata/2.3": {
            "d_value": 0.01001661,
            "K_max": 1.0,
            "condition": "nof",
            "phantom_flag": 0,
        },
        #"/data/zm/2026.1.12_testdata/1.15_150_580W": 0.01139,
    }

    channel_mask_1mm_nof_path = "/data/zm/Weiliukong/6.24/channel_mask/6.17nof/channel_mask.npy"
    channel_mask_nof1_path = "/data/zm/Weiliukong/6.24/channel_mask/nof1/channel_mask.npy"
    channel_mask_nof2_path = "/data/zm/Weiliukong/6.24/channel_mask/nof2/channel_mask.npy"

    source_level_data_config = {
        "/data/zm/Weiliukong/6.17/160_nof_data": {
            "d_value": 0.04504,
            "K_max": 0.617247,
            "condition": "nof",
            "sub_condition": "1mm_nof",
            "phantom_flag": 0,
            "split_group": "train_val",
            "quality": "valid",
            "use_for_training": True,
            "channel_mask_enabled": True,
            "channel_mask_path": channel_mask_1mm_nof_path,
        },
        "/data/zm/Weiliukong/6.17/160_withf_data": {
            "d_value": 0.024,
            "K_max": 0.440848,
            "condition": "withf",
            "sub_condition": "1mm_withf",
            "phantom_flag": 1,
            "split_group": "train_val",
            "quality": "valid",
            "use_for_training": True,
            "channel_mask_enabled": True,
            "channel_mask_path": channel_mask_1mm_nof_path,
        },
        "/data/zm/Weiliukong/6.24/withf1": {
            "d_value": 0.02666,
            "K_max": 0.432869,
            "condition": "withf",
            "sub_condition": "withf1",
            "phantom_flag": 1,
            "split_group": "train_val",
            "quality": "valid",
            "use_for_training": True,
            "channel_mask_enabled": True,
            "channel_mask_path": channel_mask_nof1_path,
        },
        "/data/zm/Weiliukong/6.24/nof1andBD": {
            "d_value": 0.0378,
            "K_max": 0.608991,
            "condition": "nof",
            "sub_condition": "nof1",
            "phantom_flag": 0,
            "split_group": "evaluate",
            "quality": "valid",
            "use_for_training": False,
            "channel_mask_enabled": True,
            "channel_mask_path": channel_mask_nof1_path,
        },
        "/data/zm/Weiliukong/6.24/withf2": {
            "d_value": 0.023138,
            "K_max": 0.411124,
            "condition": "withf",
            "sub_condition": "withf2",
            "phantom_flag": 1,
            "split_group": "evaluate",
            "quality": "valid",
            "use_for_training": False,
            "channel_mask_enabled": True,
            "channel_mask_path": channel_mask_nof2_path,
        },
        "/data/zm/Weiliukong/6.24/nof2andBD": {
            "d_value": 0.04504,
            "K_max": 0.547539,
            "condition": "nof",
            "sub_condition": "nof2",
            "phantom_flag": 0,
            "split_group": "excluded",
            "quality": "suspect_speckle_anisotropy",
            "use_for_training": False,
            "channel_mask_enabled": True,
            "channel_mask_path": channel_mask_nof2_path,
        },
    }

    if data_split_mode == "velocity":
        train_val_env_config, evaluate_env_config, excluded_env_config = split_config_by_group(
            source_level_data_config,
            train_val_group=args.train_val_split_group,
            evaluate_group=args.evaluate_split_group,
            excluded_group=args.exclude_split_group,
            enabled=bool(args.enable_split_group),
        )
        train_env_config = train_val_env_config
        val_env_config = train_val_env_config
        train_include_velocities = args.train_velocities
        val_include_velocities = args.val_velocities
    else:
        source_level_data_config = {**legacy_train_env_config, **legacy_val_env_config}
        train_env_config = legacy_train_env_config
        val_env_config = legacy_val_env_config
        evaluate_env_config = {}
        excluded_env_config = {}
        train_include_velocities = None
        val_include_velocities = None
    data_source_split_rows = build_data_source_split_rows(
        source_level_data_config,
        train_val_group=args.train_val_split_group,
        evaluate_group=args.evaluate_split_group,
        excluded_group=args.exclude_split_group,
    )

    mask_path = None if args.disable_mask else args.mask_path
    model_weights_path = "/data/zm/Weiliukong/6.24/Train_result/model/best_blood_flow_model.pth"
    loss_curve_path = "/data/zm/Weiliukong/6.24/Train_result/loss_curve"
    report_dir = "/data/zm/Weiliukong/6.24/Train_result/report"
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(report_dir, f"train_cross_{report_timestamp}.md")
    best_val_predictions_path = os.path.join(report_dir, "best_val_predictions.csv")
    best_val_condition_velocity_metrics_path = os.path.join(report_dir, "best_val_condition_velocity_metrics.csv")
    best_val_sub_condition_velocity_metrics_path = os.path.join(report_dir, "best_val_sub_condition_velocity_metrics.csv")
    evaluate_predictions_path = os.path.join(report_dir, "evaluate_predictions.csv")
    evaluate_condition_velocity_metrics_path = os.path.join(report_dir, "evaluate_condition_velocity_metrics.csv")
    evaluate_sub_condition_velocity_metrics_path = os.path.join(report_dir, "evaluate_sub_condition_velocity_metrics.csv")

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
        include_velocities=train_include_velocities,
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
        include_velocities=val_include_velocities,
    )
    evaluate_ds = None
    if evaluate_env_config:
        evaluate_ds = FlexibleBloodFlowDataset(
            evaluate_env_config,
            mask_path=mask_path,
            T=1,
            seq_len=base_total_steps,
            dt_us=dt_us,
            max_velocity=max_velocity,
            event_norm_mode=event_norm_mode,
            event_norm_stats=train_event_norm_stats,
            event_norm_reference_mean=train_event_norm_stats["reference_mean_events_per_sample"],
            event_norm_clip=event_norm_clip,
            event_intensity_jitter_range=None,
            return_metadata=True,
            include_velocities=None,
        )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Empty dataset after building train/val splits. "
            f"data_split_mode={data_split_mode}, "
            f"train_samples={len(train_ds)}, val_samples={len(val_ds)}, "
            f"train_env_config={train_env_config}, val_env_config={val_env_config}, "
            f"train_include_velocities={train_include_velocities}, val_include_velocities={val_include_velocities}. "
            "Check that paths exist, CSV filenames contain parseable velocities, selected velocities exist, "
            "and the hardcoded event ROI rows 100:200 / cols 0:1200 contains events."
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
        candidate_model = SNN_CNN_Hybrid(
            in_channels=1,
            max_velocity=max_velocity,
            use_beta_conditioning=use_beta_conditioning,
            use_bounded_scatter=use_bounded_scatter,
            scatter_scale=scatter_scale,
        ).to(device)
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
    evaluate_loader = None
    if evaluate_ds is not None and len(evaluate_ds) > 0:
        evaluate_loader = DataLoader(
            evaluate_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=sequence_sparse_collate,
            num_workers=num_workers,
        )
    print(
        f"Dataset summary | train_samples={len(train_ds)}, "
        f"train_batches={train_sampling_plan['effective_batches']}, "
        f"val_samples={len(val_ds)}, val_batches={val_sampling_plan['effective_batches']}, "
        f"evaluate_samples={len(evaluate_ds) if evaluate_ds is not None else 0}"
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
    best_per_condition_rows = []
    best_per_sub_condition_rows = []
    best_condition_velocity_rows = []
    best_sub_condition_velocity_rows = []
    best_beta_scatter_correlation_by_condition = []
    best_beta_diagnostics = {}
    best_raw_dependency = {}
    evaluate_stats = None
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
                f"gamma={val_stats['gamma_mean']:.4f}, beta_ratio={val_stats['beta_eff_ratio_mean']:.4f}, "
                f"scatter_delta={val_stats['scatter_delta_mean']:.4f}, "
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
                best_per_condition_rows = val_stats.get("per_condition", [])
                best_beta_diagnostics = {
                    key: val_stats.get(key)
                    for key in [
                        "gamma_mean",
                        "gamma_std",
                        "gamma_min",
                        "gamma_max",
                        "beta_max_mean",
                        "beta_eff_mean",
                        "beta_eff_min",
                        "beta_eff_max",
                        "beta_eff_ratio_mean",
                        "beta_eff_ratio_std",
                        "scatter_delta_mean",
                        "scatter_delta_std",
                        "scatter_delta_min",
                        "scatter_delta_max",
                        "tau_base_mean",
                        "tau_base_min",
                        "tau_base_max",
                        "tau_eff_mean",
                        "tau_eff_min",
                        "tau_eff_max",
                        "corr_gamma_velocity",
                        "corr_scatter_delta_velocity",
                        "beta_eff_lt_beta_max",
                    ]
                }
                best_raw_dependency = val_stats["raw_dependency"]
                best_per_sub_condition_rows = val_stats.get("per_sub_condition", [])
                best_condition_velocity_rows = val_stats.get("condition_velocity", [])
                best_sub_condition_velocity_rows = val_stats.get("sub_condition_velocity", [])
                best_beta_scatter_correlation_by_condition = val_stats.get("beta_scatter_correlation_by_condition", [])
                best_epoch = epoch
                save_validation_predictions_csv(best_val_predictions_path, val_stats)
                save_condition_velocity_metrics_csv(
                    best_val_condition_velocity_metrics_path,
                    best_condition_velocity_rows,
                )
                save_sub_condition_velocity_metrics_csv(
                    best_val_sub_condition_velocity_metrics_path,
                    best_sub_condition_velocity_rows,
                )
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
                        "val_per_condition": val_stats.get("per_condition", []),
                        "val_per_sub_condition": best_per_sub_condition_rows,
                        "val_condition_velocity": best_condition_velocity_rows,
                        "val_sub_condition_velocity": best_sub_condition_velocity_rows,
                        "val_beta_scatter_correlation_by_condition": best_beta_scatter_correlation_by_condition,
                        "val_beta_diagnostics": best_beta_diagnostics,
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
                        "loss_weights": current_stage["loss_weights"],
                        "data_config": {
                            "all_sources": source_level_data_config,
                            "train": train_env_config,
                            "val": val_env_config,
                            "evaluate": evaluate_env_config,
                            "excluded": excluded_env_config,
                        },
                        "beta_conditioning_config": {
                            "use_beta_conditioning": use_beta_conditioning,
                            "use_bounded_scatter": use_bounded_scatter,
                            "scatter_scale": scatter_scale,
                            "tau_base_anchor_weight": float(args.tau_base_anchor_weight),
                            "scatter_delta_reg_weight": float(args.scatter_delta_reg_weight),
                            "K_max_default": 1.0,
                            "beta_eps": 1e-8,
                        },
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
                        "data_split_config": {
                            "data_split_mode": data_split_mode,
                            "enable_split_group": bool(args.enable_split_group),
                            "train_val_split_group": args.train_val_split_group,
                            "evaluate_split_group": args.evaluate_split_group,
                            "exclude_split_group": args.exclude_split_group,
                            "train_val_config_keys": list(train_env_config.keys()),
                            "evaluate_config_keys": list(evaluate_env_config.keys()),
                            "excluded_config_keys": list(excluded_env_config.keys()),
                            "train_include_velocities": train_include_velocities,
                            "val_include_velocities": val_include_velocities,
                        },
                        "mask_config": {
                            "mask_path": mask_path,
                            "mask_enabled": mask_path is not None,
                            "channel_mask_summary": collect_channel_mask_summary_rows(train_ds, val_ds, evaluate_ds),
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
                "best_val_condition_velocity_metrics_path": best_val_condition_velocity_metrics_path,
                "best_val_sub_condition_velocity_metrics_path": best_val_sub_condition_velocity_metrics_path,
                "evaluate_predictions_path": evaluate_predictions_path,
                "evaluate_condition_velocity_metrics_path": evaluate_condition_velocity_metrics_path,
                "evaluate_sub_condition_velocity_metrics_path": evaluate_sub_condition_velocity_metrics_path,
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
                "mask_path": mask_path,
                "mask_enabled": mask_path is not None,
                "data_split_mode": data_split_mode,
                "enable_split_group": bool(args.enable_split_group),
                "train_val_split_group": args.train_val_split_group,
                "evaluate_split_group": args.evaluate_split_group,
                "exclude_split_group": args.exclude_split_group,
                "train_include_velocities": train_include_velocities,
                "val_include_velocities": val_include_velocities,
                "use_beta_conditioning": use_beta_conditioning,
                "use_bounded_scatter": use_bounded_scatter,
                "scatter_scale": scatter_scale,
                "tau_base_anchor_weight": float(args.tau_base_anchor_weight),
                "scatter_delta_reg_weight": float(args.scatter_delta_reg_weight),
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
                "evaluate_env_config": evaluate_env_config,
                "excluded_env_config": excluded_env_config,
                "data_source_split_rows": data_source_split_rows,
                "channel_mask_summary_rows": collect_channel_mask_summary_rows(train_ds, val_ds, evaluate_ds),
                "train_ds": train_ds,
                "val_ds": val_ds,
                "evaluate_ds": evaluate_ds,
                "smoke_test": smoke_test_stats,
                "best_per_velocity": best_per_velocity_rows,
                "best_per_condition": best_per_condition_rows,
                "best_per_sub_condition": best_per_sub_condition_rows,
                "best_condition_velocity": best_condition_velocity_rows,
                "best_sub_condition_velocity": best_sub_condition_velocity_rows,
                "best_beta_scatter_correlation_by_condition": best_beta_scatter_correlation_by_condition,
                "best_beta_diagnostics": best_beta_diagnostics,
                "best_raw_dependency": best_raw_dependency,
                "evaluate_stats": evaluate_stats,
            }
            write_training_report(report_path, run_info, epoch_records)
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("Training interrupted by user.")

    if train_loss_history and plt is not None:
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
    elif train_loss_history:
        print("WARNING: matplotlib is not installed; skipping loss curve plot.")

    if bool(args.run_evaluate_after_train) and evaluate_loader is not None and best_epoch >= 0:
        print("Running independent evaluate split with best checkpoint...")
        try:
            checkpoint = torch.load(model_weights_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        evaluate_stats = run_epoch(
            model,
            evaluate_loader,
            None,
            device,
            base_total_steps,
            base_block_size,
            snn_bin_size,
            snn_input_scale_mode,
            base_dt_us,
            spatial_shape,
            patch_shape,
            best_epoch,
            "Evaluate",
            max_batches=None,
            loss_weights=main_loss_weights,
            gradient_clip_max_norm=gradient_clip_max_norm,
        )
        save_validation_predictions_csv(evaluate_predictions_path, evaluate_stats)
        save_condition_velocity_metrics_csv(
            evaluate_condition_velocity_metrics_path,
            evaluate_stats.get("condition_velocity", []),
        )
        save_sub_condition_velocity_metrics_csv(
            evaluate_sub_condition_velocity_metrics_path,
            evaluate_stats.get("sub_condition_velocity", []),
        )

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
        "best_val_condition_velocity_metrics_path": best_val_condition_velocity_metrics_path,
        "best_val_sub_condition_velocity_metrics_path": best_val_sub_condition_velocity_metrics_path,
        "evaluate_predictions_path": evaluate_predictions_path,
        "evaluate_condition_velocity_metrics_path": evaluate_condition_velocity_metrics_path,
        "evaluate_sub_condition_velocity_metrics_path": evaluate_sub_condition_velocity_metrics_path,
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
        "mask_path": mask_path,
        "mask_enabled": mask_path is not None,
        "data_split_mode": data_split_mode,
        "enable_split_group": bool(args.enable_split_group),
        "train_val_split_group": args.train_val_split_group,
        "evaluate_split_group": args.evaluate_split_group,
        "exclude_split_group": args.exclude_split_group,
        "train_include_velocities": train_include_velocities,
        "val_include_velocities": val_include_velocities,
        "use_beta_conditioning": use_beta_conditioning,
        "use_bounded_scatter": use_bounded_scatter,
        "scatter_scale": scatter_scale,
        "tau_base_anchor_weight": float(args.tau_base_anchor_weight),
        "scatter_delta_reg_weight": float(args.scatter_delta_reg_weight),
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
        "evaluate_env_config": evaluate_env_config,
        "excluded_env_config": excluded_env_config,
        "data_source_split_rows": data_source_split_rows,
        "channel_mask_summary_rows": collect_channel_mask_summary_rows(train_ds, val_ds, evaluate_ds),
        "train_ds": train_ds,
        "val_ds": val_ds,
        "evaluate_ds": evaluate_ds,
        "smoke_test": smoke_test_stats,
        "best_per_velocity": best_per_velocity_rows,
        "best_per_condition": best_per_condition_rows,
        "best_per_sub_condition": best_per_sub_condition_rows,
        "best_condition_velocity": best_condition_velocity_rows,
        "best_sub_condition_velocity": best_sub_condition_velocity_rows,
        "best_beta_scatter_correlation_by_condition": best_beta_scatter_correlation_by_condition,
        "best_beta_diagnostics": best_beta_diagnostics,
        "best_raw_dependency": best_raw_dependency,
        "evaluate_stats": evaluate_stats,
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
