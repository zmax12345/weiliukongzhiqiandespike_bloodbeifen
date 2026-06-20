import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    np = None
    NP_IMPORT_ERROR = exc
else:
    NP_IMPORT_ERROR = None
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    torch = None
    F = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import MinkowskiEngine as ME
except ImportError as exc:
    ME = None
    ME_IMPORT_ERROR = exc
else:
    ME_IMPORT_ERROR = None

try:
    from dense_block_manager import DenseBlockManager
    from model import SNN_CNN_Hybrid
except ImportError as exc:
    DenseBlockManager = None
    SNN_CNN_Hybrid = None
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None


EPS = 1e-8
FILENAME_RE = re.compile(
    r"^(?P<threshold>150)_(?P<phantom>nof|withf)_(?P<velocity>\d+(?:\.\d+)?)_clip\.csv$",
    re.IGNORECASE,
)
ALLOWED_VELOCITIES = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
LOSS_KEYS = [
    "final_velocity_loss",
    "tau_log_loss",
    "rank_loss",
    "final_var_loss",
    "v_aux_loss",
    "tau_delta_reg_loss",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Pilot training for threshold150 microfluidic event data.")
    parser.add_argument("--data-root", default="/data/zm/Weiliukong/5.30/data/threshold150")
    parser.add_argument("--out-dir", default="/data/zm/Weiliukong/5.30/train_microfluidic_pilot")
    parser.add_argument("--roi", nargs=4, type=int, default=[205, 450, 90, 780], metavar=("row_start", "col_start", "height", "width"))
    parser.add_argument("--phantom-mode", choices=["nof", "withf", "both"], default="both")
    parser.add_argument("--split-mode", choices=["holdout_velocity", "phantom_transfer", "within_condition_holdout_velocity"], default="holdout_velocity")
    parser.add_argument("--val-velocities", nargs="+", type=float, default=[0.8, 1.5])
    parser.add_argument("--train-phantom", choices=["nof", "withf"], default="nof")
    parser.add_argument("--val-phantom", choices=["nof", "withf"], default="withf")
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--base-dt-us", type=float, default=20.0)
    parser.add_argument("--snn-bin-size", type=int, default=10)
    parser.add_argument("--base-block-size", type=int, default=400)
    parser.add_argument("--snn-input-scale-mode", choices=["sqrt", "mean", "none"], default="sqrt")
    parser.add_argument("--max-windows-per-file", type=int, default=20)
    parser.add_argument("--pixel-size-um", type=float, default=2.1)
    parser.add_argument("--channel-width-um", type=float, default=150.0)
    parser.add_argument("--speckle-size-px-nof", type=float, default=2.764296)
    parser.add_argument("--speckle-size-px-withf", type=float, default=0.988712)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-max-norm", type=float, default=1.0)
    parser.add_argument("--patch-shape", nargs=2, type=int, default=[30, 65])
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def write_csv(path, rows, preferred=None):
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = preferred or []
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_file_metadata(path):
    match = FILENAME_RE.match(Path(path).name.lower())
    if not match:
        raise ValueError(f"Filename does not match strict threshold150 pattern: {Path(path).name}")
    velocity = safe_float(match.group("velocity"))
    if not any(abs(velocity - allowed) < 1e-9 for allowed in ALLOWED_VELOCITIES):
        raise ValueError(f"Invalid velocity {velocity}; allowed={ALLOWED_VELOCITIES}")
    return {
        "threshold": "150",
        "phantom": match.group("phantom"),
        "velocity": velocity,
    }


def canonical_column(name):
    key = str(name).strip().lower()
    if key in {"row", "r"}:
        return "row"
    if key in {"col", "column", "c"}:
        return "col"
    if key in {"t_in", "tin", "t", "timestamp", "time", "ts", "time_us"}:
        return "time_us"
    if key in {"t_off", "toff"}:
        return "t_off"
    return ""


def load_csv(path):
    if pd is not None:
        df = pd.read_csv(path)
        hits = sum(1 for col in df.columns if canonical_column(col))
        numeric_header = 0
        for col in df.columns:
            try:
                float(col)
                numeric_header += 1
            except (TypeError, ValueError):
                pass
        if hits < 2 and numeric_header >= 2:
            df = pd.read_csv(path, header=None, names=["row", "col", "t_in", "t_off"])
        return df
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if arr.dtype.names:
        return {name: np.asarray(arr[name]) for name in arr.dtype.names}
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    names = ["row", "col", "t_in", "t_off"][: arr.shape[1]]
    return {name: arr[:, idx] for idx, name in enumerate(names)}


def data_columns(data):
    return list(data.columns) if pd is not None and hasattr(data, "columns") else list(data.keys())


def get_column(data, col):
    if pd is not None and hasattr(data, "columns"):
        return pd.to_numeric(data[col], errors="coerce").to_numpy()
    return np.asarray(data[col], dtype=np.float64)


def read_event_csv(path):
    data = load_csv(path)
    mapping = {col: canonical_column(col) for col in data_columns(data)}
    row_col = next((col for col, kind in mapping.items() if kind == "row"), None)
    col_col = next((col for col, kind in mapping.items() if kind == "col"), None)
    time_col = next((col for col, kind in mapping.items() if kind == "time_us"), None)
    if row_col is None or col_col is None or time_col is None:
        raise ValueError(f"Cannot find row/col/t_in columns in {path}. Columns={data_columns(data)}")
    row = get_column(data, row_col)
    col = get_column(data, col_col)
    time_us = get_column(data, time_col)
    finite = np.isfinite(row) & np.isfinite(col) & np.isfinite(time_us)
    row = row[finite].astype(np.int64)
    col = col[finite].astype(np.int64)
    time_us = time_us[finite].astype(np.float64)
    valid = (row >= 0) & (col >= 0)
    return time_us[valid], row[valid], col[valid]


def select_window_starts(time_us, window_us, max_windows):
    if time_us.size == 0:
        return []
    t_min = float(np.min(time_us))
    t_max = float(np.max(time_us))
    if t_max - t_min < window_us:
        return []
    start_max = t_max - window_us
    total = int(math.floor((start_max - t_min) / window_us)) + 1
    if total <= max_windows:
        return [t_min + idx * window_us for idx in range(total)]
    return np.linspace(t_min, start_max, max_windows).tolist()


def build_window_sequence(time_us, row, col, start_us, args):
    row_start, col_start, height, width = args.roi
    window_us = args.window_ms * 1000.0
    end_us = start_us + window_us
    keep = (
        (time_us >= start_us)
        & (time_us < end_us)
        & (row >= row_start)
        & (row < row_start + height)
        & (col >= col_start)
        & (col < col_start + width)
    )
    t = time_us[keep]
    r = row[keep] - row_start
    c = col[keep] - col_start
    total_steps = int(window_us // args.base_dt_us)
    frame_map = defaultdict(list)
    if t.size:
        frame_idx = np.floor((t - start_us) / args.base_dt_us).astype(np.int64)
        valid = (frame_idx >= 0) & (frame_idx < total_steps)
        frame_idx = frame_idx[valid]
        r = r[valid]
        c = c[valid]
        for f_idx, rr, cc in zip(frame_idx, r, c):
            frame_map[int(f_idx)].append((int(rr), int(cc)))

    sequence = []
    raw_total_events = 0
    for f_idx in range(total_steps):
        coords_list = frame_map.get(f_idx, [])
        if coords_list:
            coords_np = np.unique(np.asarray(coords_list, dtype=np.int32), axis=0)
            coords = torch.as_tensor(coords_np, dtype=torch.int32)
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            raw_total_events += int(coords.shape[0])
        else:
            coords = torch.empty((0, 2), dtype=torch.int32)
            feats = torch.empty((0, 1), dtype=torch.float32)
        sequence.append((coords, feats))

    keep_fraction = float(np.sum(keep) / max(len(time_us[(time_us >= start_us) & (time_us < end_us)]), 1))
    diag = {}
    if r.size:
        diag = {
            "row_com": float(np.mean(r)),
            "col_com": float(np.mean(c)),
            "row_std": float(np.std(r)),
            "col_std": float(np.std(c)),
        }
    else:
        diag = {"row_com": float("nan"), "col_com": float("nan"), "row_std": float("nan"), "col_std": float("nan")}
    return sequence, raw_total_events, keep_fraction, diag


def discover_windows(args):
    files = sorted(Path(args.data_root).glob("*.csv"))
    windows = []
    failed = []
    file_id = 0
    for path in files:
        try:
            meta = parse_file_metadata(path)
            if args.phantom_mode != "both" and meta["phantom"] != args.phantom_mode:
                continue
            time_us, row, col = read_event_csv(path)
            starts = select_window_starts(time_us, args.window_ms * 1000.0, args.max_windows_per_file)
            if not starts:
                failed.append({"file_path": str(path), "error": "not_enough_time_for_one_window"})
                continue
            for win_idx, start in enumerate(starts):
                sequence, raw_total, keep_fraction, spatial_diag = build_window_sequence(time_us, row, col, start, args)
                windows.append(
                    {
                        "sequence": sequence,
                        "velocity": meta["velocity"],
                        "d_value": 1.0,
                        "phantom": meta["phantom"],
                        "threshold": meta["threshold"],
                        "file_id": file_id,
                        "file_name": path.name,
                        "file_path": str(path),
                        "window_index": win_idx,
                        "window_start_us": float(start),
                        "window_end_us": float(start + args.window_ms * 1000.0),
                        "raw_total_events": raw_total,
                        "roi_keep_fraction": keep_fraction,
                        **spatial_diag,
                    }
                )
            file_id += 1
        except Exception as exc:
            failed.append({"file_path": str(path), "error": str(exc)})
    return windows, failed


def split_windows(windows, args):
    val_velocities = set(round(v, 6) for v in args.val_velocities)
    train, val = [], []
    for sample in windows:
        velocity = round(float(sample["velocity"]), 6)
        phantom = sample["phantom"]
        if args.split_mode == "holdout_velocity":
            (val if velocity in val_velocities else train).append(sample)
        elif args.split_mode == "phantom_transfer":
            if phantom == args.train_phantom:
                train.append(sample)
            elif phantom == args.val_phantom:
                val.append(sample)
        elif args.split_mode == "within_condition_holdout_velocity":
            condition = args.phantom_mode if args.phantom_mode in {"nof", "withf"} else args.train_phantom
            if phantom != condition:
                continue
            (val if velocity in val_velocities else train).append(sample)
    return train, val


class MicrofluidicWindowDataset(Dataset):
    def __init__(self, samples, spatial_shape):
        self.samples = list(samples)
        self.spatial_shape = tuple(spatial_shape)
        self.source_sample_counts = dict()
        self.velocity_sample_counts = defaultdict(int)
        self.phantom_sample_counts = defaultdict(int)
        for sample in self.samples:
            self.velocity_sample_counts[float(sample["velocity"])] += 1
            self.phantom_sample_counts[sample["phantom"]] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        env_map = torch.zeros((1, *self.spatial_shape), dtype=torch.float32)
        source_id = 0 if sample["phantom"] == "nof" else 1
        metadata = {k: v for k, v in sample.items() if k != "sequence"}
        return sample["sequence"], float(sample["velocity"]), float(sample["d_value"]), env_map, source_id, metadata


def sequence_sparse_collate(batch):
    seq_len = len(batch[0][0])
    batched_seq = []
    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in batch]
        feats_t = [sample[0][t][1] for sample in batch]
        b_coords, b_feats = ME.utils.sparse_collate(coords_t, feats_t)
        batched_seq.append((b_coords, b_feats))
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    d_values = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
    env_maps = torch.stack([sample[3] for sample in batch], dim=0)
    source_ids = torch.tensor([sample[4] for sample in batch], dtype=torch.long)
    metadata = [sample[5] for sample in batch]
    return batched_seq, labels, d_values, env_maps, source_ids, metadata


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")
    return safe_pearson(rankdata(x), rankdata(y))


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def compute_scalar_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan")
    y_true, y_pred = y_true[finite], y_pred[finite]
    err = np.abs(y_true - y_pred)
    return float(err.mean()), float(np.sqrt(np.mean((y_true - y_pred) ** 2))), float(np.mean(err / np.maximum(np.abs(y_true), EPS)) * 100.0)


def pairwise_rank_accuracy(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[valid], target[valid]
    if pred.size < 2:
        return float("nan")
    td = target.reshape(-1, 1) - target.reshape(1, -1)
    pd = pred.reshape(-1, 1) - pred.reshape(1, -1)
    mask = td > 0
    return float(np.mean(pd[mask] > 0)) if np.any(mask) else float("nan")


def pairwise_ranking_loss(pred, target, margin=0.12):
    target_diff = target.view(-1, 1) - target.view(1, -1)
    pred_diff = pred.view(-1, 1) - pred.view(1, -1)
    valid = target_diff > 0
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    return F.relu(margin - pred_diff[valid]).mean()


def compute_loss(output, d_values, y_true, weights):
    tau_pred = output["tau_pred"]
    log_tau_pred = output["log_tau_pred"]
    v_final = d_values / torch.clamp(tau_pred, min=1e-8)
    tau_target = d_values / torch.clamp(y_true, min=1e-8)
    loss_final_velocity = F.smooth_l1_loss(v_final, y_true)
    loss_tau_log = F.smooth_l1_loss(log_tau_pred, torch.log(torch.clamp(tau_target, min=1e-8)))
    loss_rank = pairwise_ranking_loss(v_final, y_true, margin=weights["rank_margin"])
    loss_final_var = F.relu(weights["pred_std_fraction"] * y_true.std(unbiased=False) - v_final.std(unbiased=False))
    v_aux = output.get("v_pred")
    loss_v_aux = F.smooth_l1_loss(v_aux, y_true) if v_aux is not None else y_true.new_tensor(0.0)
    log_tau_delta = output.get("log_tau_delta")
    loss_tau_delta_reg = log_tau_delta.pow(2).mean() if log_tau_delta is not None else y_true.new_tensor(0.0)
    components = {
        "final_velocity_loss": loss_final_velocity,
        "tau_log_loss": loss_tau_log,
        "rank_loss": loss_rank,
        "final_var_loss": loss_final_var,
        "v_aux_loss": loss_v_aux,
        "tau_delta_reg_loss": loss_tau_delta_reg,
    }
    total = (
        weights["final_velocity"] * loss_final_velocity
        + weights["tau_log"] * loss_tau_log
        + weights["rank"] * loss_rank
        + weights["final_var"] * loss_final_var
        + weights["v_aux"] * loss_v_aux
        + weights["tau_delta_reg"] * loss_tau_delta_reg
    )
    return total, components, v_final


def per_velocity_stats(y_true, y_pred):
    rows = []
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    for velocity in sorted(np.unique(y_true[np.isfinite(y_true)])):
        mask = np.isclose(y_true, velocity) & np.isfinite(y_pred)
        if not np.any(mask):
            continue
        mae, rmse, mape = compute_scalar_metrics(y_true[mask], y_pred[mask])
        pred_mean = float(np.mean(y_pred[mask]))
        rows.append(
            {
                "velocity": float(velocity),
                "samples": int(mask.sum()),
                "pred_mean": pred_mean,
                "pred_std": float(np.std(y_pred[mask])),
                "bias": pred_mean - float(velocity),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )
    return rows


def run_epoch(model, loader, optimizer, device, args, loss_weights, split_name):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    comp_sums = {key: 0.0 for key in LOSS_KEYS}
    all_true, all_pred, all_aux, all_tau, all_log_tau = [], [], [], [], []
    diag = defaultdict(float)
    batches = 0
    spatial_shape = (args.roi[2], args.roi[3])
    patch_shape = tuple(args.patch_shape)
    for batch in loader:
        x_seq_sparse_data, y_true, d_values, env_maps, source_ids, metadata = batch
        y_true = y_true.to(device)
        d_values = d_values.to(device)
        manager = DenseBlockManager(x_seq_sparse_data, batch_size=y_true.shape[0], spatial_shape=spatial_shape, patch_shape=patch_shape)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            output = model(
                dataloader_or_generator=manager,
                base_total_steps=int(args.window_ms * 1000 // args.base_dt_us),
                base_block_size=args.base_block_size,
                snn_bin_size=args.snn_bin_size,
                snn_input_scale_mode=args.snn_input_scale_mode,
                base_dt_us=args.base_dt_us,
            )
            loss, components, v_final = compute_loss(output, d_values, y_true, loss_weights)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_max_norm)
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        for key in LOSS_KEYS:
            comp_sums[key] += float(components[key].detach().cpu())
        batches += 1
        all_true.extend(y_true.detach().cpu().numpy().tolist())
        all_pred.extend(v_final.detach().cpu().numpy().tolist())
        all_aux.extend(output.get("v_pred", torch.full_like(v_final, float("nan"))).detach().cpu().numpy().tolist())
        all_tau.extend(output["tau_pred"].detach().cpu().numpy().tolist())
        all_log_tau.extend(output["log_tau_pred"].detach().cpu().numpy().tolist())
        for i in (1, 2, 3):
            feat = output[f"snn_feat_{i}"].detach()
            diag[f"feat{i}_std"] += float(feat.std(unbiased=False).cpu())
            diag[f"layer{i}_spike_rate"] += float(output.get(f"layer{i}_spike_rate", float("nan")))
        diag["cnn_embedding_std"] += float(output["cnn_embedding"].detach().std(unbiased=False).cpu())

    if batches == 0:
        raise RuntimeError(f"{split_name} loader produced no batches.")
    mae, rmse, mape = compute_scalar_metrics(all_true, all_pred)
    per_vel = per_velocity_stats(all_true, all_pred)
    bin_mae_max = max([r["mae"] for r in per_vel], default=float("nan"))
    stats = {
        "loss": total_loss / batches,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pearson": safe_pearson(all_true, all_pred),
        "spearman": safe_spearman(all_true, all_pred),
        "rank_accuracy": pairwise_rank_accuracy(all_pred, all_true),
        "pred_std": float(np.std(all_pred)),
        "pred_min": float(np.min(all_pred)),
        "pred_max": float(np.max(all_pred)),
        "tau_min": float(np.min(all_tau)),
        "tau_max": float(np.max(all_tau)),
        "log_tau_min": float(np.min(all_log_tau)),
        "log_tau_max": float(np.max(all_log_tau)),
        "per_velocity": per_vel,
        "bin_mae_max": bin_mae_max,
        "predictions": all_pred,
        "targets": all_true,
        "v_aux": all_aux,
    }
    for key in LOSS_KEYS:
        stats[key] = comp_sums[key] / batches
    for key, value in diag.items():
        stats[key] = value / batches
    return stats


def is_better(val_stats, best_stats):
    if not np.isfinite(val_stats["mae"]) or val_stats["pred_std"] < 1e-8:
        return False
    if best_stats is None:
        return True
    if val_stats["mae"] < best_stats["mae"] - 1e-9:
        return True
    if val_stats["mae"] > best_stats["mae"] + 1e-9:
        return False
    for key, higher in (("bin_mae_max", False), ("rank_accuracy", True), ("pred_std", True), ("loss", False)):
        cur, best = val_stats.get(key, float("nan")), best_stats.get(key, float("nan"))
        if higher and cur > best:
            return True
        if (not higher) and cur < best:
            return True
    return False


def spatial_drift_summary(samples, split_name):
    grouped = defaultdict(list)
    for s in samples:
        grouped[(split_name, s["phantom"], float(s["velocity"]))].append(s)
    rows = []
    for (split, phantom, velocity), items in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        row = {"split": split, "phantom": phantom, "velocity": velocity, "samples": len(items)}
        for key in ("row_com", "col_com", "row_std", "col_std"):
            vals = np.asarray([safe_float(item.get(key)) for item in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{key}_std"] = float(np.std(vals)) if vals.size else float("nan")
        rows.append(row)
    return rows


def save_loss_curve(path, epoch_rows):
    if plt is None or not epoch_rows:
        return
    plt.figure(figsize=(9, 5))
    plt.plot([r["epoch"] for r in epoch_rows], [r["train_loss"] for r in epoch_rows], label="train")
    plt.plot([r["epoch"] for r in epoch_rows], [r["val_loss"] for r in epoch_rows], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    if not rows:
        lines.append("| " + " | ".join(["-"] * len(headers)) + " |")
        return "\n".join(lines)
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                cells.append(f"{v:.6g}" if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path, args, train_samples, val_samples, failed, epoch_rows, best_stats, best_epoch, per_velocity, spatial_rows):
    nof_um = args.speckle_size_px_nof * args.pixel_size_um
    withf_um = args.speckle_size_px_withf * args.pixel_size_um
    row_start, col_start, height, width = args.roi
    avg_keep = np.mean([s["roi_keep_fraction"] for s in train_samples + val_samples]) if train_samples or val_samples else float("nan")
    lines = [
        "# Microfluidic Pilot Training Report",
        "",
        "## Terminology",
        "",
        "- threshold = event camera threshold.",
        "- nof = no phantom, no scattering phantom.",
        "- withf = with phantom, scattering phantom present.",
        "- ROI = region of interest.",
        "- pilot training = small-sample preliminary training.",
        "- tau = decorrelation time.",
        "",
        "## Dataset Summary",
        "",
        f"- data_root: `{args.data_root}`",
        f"- train windows: `{len(train_samples)}`",
        f"- val windows: `{len(val_samples)}`",
        f"- failed files: `{len(failed)}`",
        f"- average ROI keep fraction: `{avg_keep:.6f}`",
        "",
        "## ROI Setting",
        "",
        f"- row range: `{row_start}-{row_start + height - 1}`",
        f"- col range: `{col_start}-{col_start + width - 1}`",
        f"- ROI height: `{height}`",
        f"- ROI width: `{width}`",
        "",
        "## Physical Calibration",
        "",
        f"- effective_pixel_size_um: `{args.pixel_size_um}`",
        f"- channel_width_um: `{args.channel_width_um}`",
        f"- speckle_size_um_nof: `{nof_um:.6f}`",
        f"- speckle_size_um_withf: `{withf_um:.6f}`",
        "- pixel_size_um comes from event accumulation approximate calibration using the known 150 um channel width; it is not a strict optical ruler calibration.",
        "- pixel_size_um is not used as model input.",
        "- d_value is fixed to 1.0 for pilot training; this is a temporary proportional factor.",
        "",
        "## Time Slicing Config",
        "",
        f"- window_ms: `{args.window_ms}`",
        f"- base_dt_us: `{args.base_dt_us}`",
        f"- snn_bin_size: `{args.snn_bin_size}`",
        f"- snn_step_us: `{args.base_dt_us * args.snn_bin_size}`",
        f"- snn_steps: `{int(args.window_ms * 1000 // (args.base_dt_us * args.snn_bin_size))}`",
        f"- snn_input_scale_mode: `{args.snn_input_scale_mode}`",
        "",
        "## Split Mode",
        "",
        f"- phantom_mode: `{args.phantom_mode}`",
        f"- split_mode: `{args.split_mode}`",
        f"- val_velocities: `{args.val_velocities}`",
        f"- train_phantom: `{args.train_phantom}`",
        f"- val_phantom: `{args.val_phantom}`",
        "- No window-level random split is used; split is based on velocity and/or phantom condition.",
        "",
        "## Model Config",
        "",
        "- Backbone: Legacy SNNEncoder -> SNNFeatureCNNDecoder -> sample-level tau_pred.",
        "- Final prediction: `v_final = d_value / tau_pred`.",
        "- No beta head, no scattering head, no raw direct velocity head.",
        "- row_com/col_com are diagnostics only and are not model inputs.",
        "",
        "## Best Validation Result",
        "",
        f"- best_epoch: `{best_epoch}`",
        f"- val_final_mae: `{best_stats.get('mae', float('nan')):.6f}`",
        f"- val_rmse: `{best_stats.get('rmse', float('nan')):.6f}`",
        f"- val_mape: `{best_stats.get('mape', float('nan')):.2f}%`",
        f"- pearson: `{best_stats.get('pearson', float('nan')):.6f}`",
        f"- spearman: `{best_stats.get('spearman', float('nan')):.6f}`",
        f"- rank_accuracy: `{best_stats.get('rank_accuracy', float('nan')):.6f}`",
        f"- pred_std: `{best_stats.get('pred_std', float('nan')):.6f}`",
        "",
        "## Per-Velocity Result",
        "",
        markdown_table(["velocity", "samples", "pred_mean", "pred_std", "bias", "mae", "rmse", "mape"], per_velocity),
        "",
        "## Phantom-wise Result",
        "",
        "See `val_predictions.csv` for per-window phantom labels and predictions.",
        "",
        "## Spatial Drift Diagnostic",
        "",
        "If model performance is good but row_com/col_com varies strongly with velocity, beware of spatial shortcut.",
        "",
        markdown_table(["split", "phantom", "velocity", "samples", "row_com_mean", "col_com_mean", "row_std_mean", "col_std_mean"], spatial_rows),
        "",
        "## Final Conclusion",
        "",
        "- This run is a pilot training experiment to test learnability, not a final generalized model.",
        "- If validation prediction collapses to a constant, current event representation may be insufficient or split too hard.",
        "- If validation works only where spatial drift is strong, shrink ROI or improve acquisition stability before full training.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    if np is None:
        raise RuntimeError(f"NumPy is required for training. Original import error: {NP_IMPORT_ERROR}")
    if torch is None or DataLoader is None:
        raise RuntimeError(f"PyTorch is required for training. Original import error: {TORCH_IMPORT_ERROR}")
    if ME is None:
        raise RuntimeError(f"MinkowskiEngine is required for sparse collation. Original import error: {ME_IMPORT_ERROR}")
    if DenseBlockManager is None or SNN_CNN_Hybrid is None:
        raise RuntimeError(f"Project model imports failed. Original import error: {MODEL_IMPORT_ERROR}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_us = args.window_ms * 1000.0
    snn_step_us = args.base_dt_us * args.snn_bin_size
    if window_us % snn_step_us != 0:
        raise ValueError("window_ms * 1000 must be divisible by base_dt_us * snn_bin_size.")
    base_total_steps = int(window_us // args.base_dt_us)
    if base_total_steps % args.base_block_size != 0 or args.base_block_size % args.snn_bin_size != 0:
        raise ValueError("base_total_steps/base_block_size/snn_bin_size must divide cleanly.")
    if args.roi[2] % args.patch_shape[0] != 0 or args.roi[3] % args.patch_shape[1] != 0:
        raise ValueError(f"patch_shape={args.patch_shape} must evenly divide ROI height/width={args.roi[2:]}.")

    all_windows, failed = discover_windows(args)
    train_windows, val_windows = split_windows(all_windows, args)
    if not train_windows or not val_windows:
        raise RuntimeError(f"Empty split: train={len(train_windows)}, val={len(val_windows)}.")
    spatial_shape = (args.roi[2], args.roi[3])
    train_ds = MicrofluidicWindowDataset(train_windows, spatial_shape)
    val_ds = MicrofluidicWindowDataset(val_windows, spatial_shape)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=sequence_sparse_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=sequence_sparse_collate, num_workers=args.num_workers)

    loss_weights = {
        "final_velocity": 1.0,
        "tau_log": 0.75,
        "rank": 0.5,
        "final_var": 0.5,
        "v_aux": 0.1,
        "tau_delta_reg": 0.001,
        "rank_margin": 0.12,
        "pred_std_fraction": 0.8,
    }
    default_schedule = [
        ("stage1_warm", 8, 1e-4),
        ("stage2_stable", 20, 5e-5),
        ("stage3_finetune", 12, 2e-5),
        ("stage4_refine", 10, 1e-5),
    ]
    schedule = []
    remaining = args.epochs
    for name, ep, lr in default_schedule:
        take = min(ep, remaining)
        if take > 0:
            schedule.append({"name": name, "epochs": take, "lr": lr})
            remaining -= take
    if remaining > 0:
        schedule.append({"name": "extra_refine", "epochs": remaining, "lr": 1e-5})

    model = SNN_CNN_Hybrid(in_channels=1, max_velocity=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=schedule[0]["lr"], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    epoch_rows = []
    best_stats = None
    best_epoch = -1
    best_predictions = []
    best_model_path = out_dir / "best_microfluidic_model.pth"
    epoch = 0
    start_time = time.time()
    for stage in schedule:
        for group in optimizer.param_groups:
            group["lr"] = stage["lr"]
        for _ in range(stage["epochs"]):
            epoch += 1
            train_stats = run_epoch(model, train_loader, optimizer, device, args, loss_weights, "train")
            val_stats = run_epoch(model, val_loader, None, device, args, loss_weights, "val")
            scheduler.step(val_stats["loss"])
            row = {
                "epoch": epoch,
                "stage": stage["name"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "val_final_mae": val_stats["mae"],
                "val_rmse": val_stats["rmse"],
                "val_mape": val_stats["mape"],
                "pearson": val_stats["pearson"],
                "spearman": val_stats["spearman"],
                "rank_accuracy": val_stats["rank_accuracy"],
                "pred_std": val_stats["pred_std"],
                "pred_min": val_stats["pred_min"],
                "pred_max": val_stats["pred_max"],
                "tau_min": val_stats["tau_min"],
                "tau_max": val_stats["tau_max"],
                "log_tau_min": val_stats["log_tau_min"],
                "log_tau_max": val_stats["log_tau_max"],
                "layer1_spike_rate": val_stats.get("layer1_spike_rate", float("nan")),
                "layer2_spike_rate": val_stats.get("layer2_spike_rate", float("nan")),
                "layer3_spike_rate": val_stats.get("layer3_spike_rate", float("nan")),
                "feat1_std": val_stats.get("feat1_std", float("nan")),
                "feat2_std": val_stats.get("feat2_std", float("nan")),
                "feat3_std": val_stats.get("feat3_std", float("nan")),
                "cnn_embedding_std": val_stats.get("cnn_embedding_std", float("nan")),
            }
            epoch_rows.append(row)
            print(
                f"Epoch {epoch:03d} {stage['name']} train_loss={train_stats['loss']:.4f} "
                f"val_mae={val_stats['mae']:.4f} pred_std={val_stats['pred_std']:.4f} "
                f"rank={val_stats['rank_accuracy']:.4f}"
            )
            if is_better(val_stats, best_stats):
                best_stats = dict(val_stats)
                best_epoch = epoch
                best_predictions = []
                for sample, target, pred, aux in zip(val_windows, val_stats["targets"], val_stats["predictions"], val_stats["v_aux"]):
                    best_predictions.append(
                        {
                            "file_name": sample["file_name"],
                            "phantom": sample["phantom"],
                            "velocity_true": target,
                            "v_final": pred,
                            "v_aux": aux,
                            "abs_error": abs(pred - target),
                            "window_start_us": sample["window_start_us"],
                            "window_end_us": sample["window_end_us"],
                        }
                    )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_val_mae": val_stats["mae"],
                        "input_config": {
                            "window_ms": args.window_ms,
                            "base_dt_us": args.base_dt_us,
                            "base_total_steps": base_total_steps,
                            "base_block_size": args.base_block_size,
                            "snn_bin_size": args.snn_bin_size,
                            "snn_step_us": snn_step_us,
                            "snn_steps": int(window_us // snn_step_us),
                            "snn_input_scale_mode": args.snn_input_scale_mode,
                            "spatial_shape": spatial_shape,
                            "patch_shape": tuple(args.patch_shape),
                        },
                        "args": vars(args),
                    },
                    best_model_path,
                )

    save_loss_curve(out_dir / "loss_curve.png", epoch_rows)
    write_csv(out_dir / "epoch_history.csv", epoch_rows)
    write_csv(out_dir / "dataset_windows.csv", [{k: v for k, v in s.items() if k != "sequence"} for s in all_windows])
    write_csv(out_dir / "val_predictions.csv", best_predictions)
    per_vel = best_stats.get("per_velocity", []) if best_stats else []
    write_csv(out_dir / "per_velocity_metrics.csv", per_vel)
    spatial_rows = spatial_drift_summary(train_windows, "train") + spatial_drift_summary(val_windows, "val")
    write_csv(out_dir / "spatial_drift_summary.csv", spatial_rows)
    write_json(
        out_dir / "train_config.json",
        {
            "args": vars(args),
            "schedule": schedule,
            "loss_weights": loss_weights,
            "failed_files": failed,
            "elapsed_seconds": time.time() - start_time,
        },
    )
    write_report(out_dir / "training_report.md", args, train_windows, val_windows, failed, epoch_rows, best_stats or {}, best_epoch, per_vel, spatial_rows)
    print(f"Saved best model: {best_model_path}")
    print(f"Saved report: {out_dir / 'training_report.md'}")


if __name__ == "__main__":
    main()
