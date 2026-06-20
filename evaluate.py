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


def load_trusted_checkpoint(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


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
            "## Notes",
            "",
            "- Final prediction is `d_values / tau_pred` from the sample-level tau head.",
            "- `v_pred` is auxiliary only; no beta, raw direct head, fusion head, or patch tau map is used.",
            "",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
    spatial_shape = (100, 368)
    patch_shape = (50, 46)
    dt_us = base_dt_us
    max_velocity = 2.0
    max_eval_batches = None
    event_norm_mode = "source_scale"
    event_norm_clip = (0.25, 4.0)

    test_data_config = {
        "/data/zm/Moshaboli/new_data/no5": 0.01978,
    }
    mask_path = "/data/zm/Moshaboli/new_data/other_data/3.0_mask (2)_hot_pixel_mask.npy"
    model_weights_path = "/data/zm/Moshaboli/new_data/Model/best_blood_flow_model.pth"
    loss_curve_dir = "/data/zm/Moshaboli/new_data/Loss_curve"
    report_dir = "/data/zm/Moshaboli/new_data/Markdown"
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generalization_output_dir = os.path.join(loss_curve_dir, f"generalization_{report_timestamp}")
    save_plot_path = os.path.join(generalization_output_dir, "generalization_evaluate.png")
    save_prediction_path = os.path.join(generalization_output_dir, "generalization_predictions.csv")
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

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_stage = checkpoint.get("stage", "unknown")
        checkpoint_event_norm_config = checkpoint.get("event_norm_config", {})
        max_velocity = checkpoint_event_norm_config.get("max_velocity", checkpoint.get("max_velocity", max_velocity))
        model = SNN_CNN_Hybrid(in_channels=1, max_velocity=max_velocity).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        checkpoint_stage = "legacy_state_dict"
        checkpoint_event_norm_config = {}
        model = SNN_CNN_Hybrid(in_channels=1, max_velocity=max_velocity).to(device)
        model.load_state_dict(checkpoint)
    model.eval()
    print(
        f"=> Loaded checkpoint metadata | stage={checkpoint_stage}, "
        f"event_norm_mode={event_norm_mode}, max_velocity={max_velocity}"
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
        for batch_idx, (x_seq_sparse_data, y_true, d_values, env_maps, source_ids) in progress_bar:
            if batch_idx >= effective_max_eval_batches:
                break
            d_values = d_values.to(device)
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
            )
            v_aux = model_output["v_pred"]
            v_final = d_values / torch.clamp(model_output["tau_pred"], min=1e-8)

            all_v_true.extend(y_true.numpy().tolist())
            all_v_final.extend(v_final.cpu().numpy().tolist())
            all_v_aux.extend(v_aux.cpu().numpy().tolist())
            all_tau_pred.extend(model_output["tau_pred"].cpu().numpy().tolist())
            all_log_tau.extend(model_output["log_tau_pred"].cpu().numpy().tolist())
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
            "max_eval_batches": max_eval_batches,
            "eval_sampling_plan": eval_sampling_plan,
            "eval_batches": len(test_loader),
            "test_ds": test_dataset,
        },
        eval_record,
    )
    print(f"=> Saved predictions to: {save_prediction_path}")
    print(f"=> Saved plot to: {save_plot_path}")
    print(f"=> Saved evaluation report to: {report_path}")


if __name__ == "__main__":
    evaluate_generalization()
