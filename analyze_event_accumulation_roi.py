import argparse
import csv
import math
import os
import re
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    np = None
    NP_IMPORT_ERROR = exc
else:
    NP_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None


EPS = 1e-8
DEFAULT_ROOTS = [
    "/data/zm/Weiliukong/5.30/data/threshold150",
    "/data/zm/Weiliukong/5.30/data/threshold200",
]
DEFAULT_OUT_DIR = "/data/zm/Weiliukong/5.30/data/henxiang"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Celex event accumulation ROI from CSV files.")
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Root folders containing event CSV files.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--max-events", type=int, default=3_000_000, help="Maximum events loaded per CSV.")
    parser.add_argument("--col-blocks", type=int, default=8, help="Number of column blocks for event statistics.")
    parser.add_argument("--row-blocks", type=int, default=4, help="Number of row blocks for event statistics.")
    parser.add_argument("--margin", type=int, default=2, help="Pixel margin added to recommended ROI bounds.")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_name(path):
    stem = Path(path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def safe_file_output_name(path):
    path = Path(path)
    parent = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return f"{parent}__{stem}"


def parse_metadata(csv_path):
    path = Path(csv_path)
    text = f"{path.parent.name}_{path.stem}".lower()
    threshold = ""
    match = re.search(r"threshold[_-]?(\d+)|thr[_-]?(\d+)", text)
    if match:
        threshold = next(group for group in match.groups() if group)

    phantom = "unknown"
    if re.search(r"no[_-]?phantom|nofangti|without[_-]?f|nof\b|no[_-]?f", text):
        phantom = "no_phantom"
    elif re.search(r"phantom|fangti|with[_-]?f|withf", text):
        phantom = "phantom"

    velocity = ""
    velocity_patterns = [
        r"(?:velocity|vel|speed|v)[_-]?(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:mm|mms|mmps|mm_s)",
    ]
    for pattern in velocity_patterns:
        match = re.search(pattern, text)
        if match:
            velocity = match.group(1)
            break
    return {"threshold": threshold, "phantom": phantom, "velocity": velocity}


def find_csv_files(roots):
    files = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Warning: root does not exist: {root}")
            continue
        files.extend(sorted(root_path.rglob("*.csv")))
    return files


def normalize_columns(columns):
    mapping = {}
    for col in columns:
        key = str(col).strip().lower()
        if key in {"row", "r", "y", "y_addr", "yaddr"}:
            mapping[col] = "row"
        elif key in {"col", "column", "c", "x", "x_addr", "xaddr"}:
            mapping[col] = "col"
        elif key in {"t", "ts", "time", "timestamp"}:
            mapping[col] = "t"
        elif key in {"p", "pol", "polarity"}:
            mapping[col] = "p"
    return mapping


def read_events_with_pandas(csv_path, max_events):
    first = pd.read_csv(csv_path, nrows=5)
    mapping = normalize_columns(first.columns)
    if {"row", "col"}.issubset(set(mapping.values())):
        usecols = [col for col, canonical in mapping.items() if canonical in {"row", "col", "t", "p"}]
        rows = []
        loaded = 0
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=500_000):
            chunk = chunk.rename(columns=mapping)
            take = min(len(chunk), max_events - loaded)
            rows.append(chunk.iloc[:take])
            loaded += take
            if loaded >= max_events:
                break
        data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row", "col"])
        return data

    data = pd.read_csv(csv_path, header=None, names=["row", "col", "t", "p"], usecols=[0, 1, 2, 3], nrows=max_events)
    return data


def read_events_with_numpy(csv_path, max_events):
    try:
        arr = np.genfromtxt(csv_path, delimiter=",", names=True, max_rows=max_events, dtype=None, encoding=None)
        names = arr.dtype.names or []
        mapping = normalize_columns(names)
        if {"row", "col"}.issubset(set(mapping.values())):
            result = {}
            for original, canonical in mapping.items():
                result[canonical] = np.asarray(arr[original])
            return result
    except Exception:
        pass
    arr = np.genfromtxt(csv_path, delimiter=",", max_rows=max_events, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {
        "row": arr[:, 0] if arr.shape[1] > 0 else np.asarray([]),
        "col": arr[:, 1] if arr.shape[1] > 1 else np.asarray([]),
        "t": arr[:, 2] if arr.shape[1] > 2 else np.full(arr.shape[0], np.nan),
        "p": arr[:, 3] if arr.shape[1] > 3 else np.full(arr.shape[0], np.nan),
    }


def read_events(csv_path, max_events):
    if pd is not None:
        data = read_events_with_pandas(csv_path, max_events)
        row = pd.to_numeric(data["row"], errors="coerce").to_numpy()
        col = pd.to_numeric(data["col"], errors="coerce").to_numpy()
        t = pd.to_numeric(data["t"], errors="coerce").to_numpy() if "t" in data else np.full(len(data), np.nan)
        p = pd.to_numeric(data["p"], errors="coerce").to_numpy() if "p" in data else np.full(len(data), np.nan)
    else:
        data = read_events_with_numpy(csv_path, max_events)
        row = np.asarray(data["row"], dtype=np.float64)
        col = np.asarray(data["col"], dtype=np.float64)
        t = np.asarray(data.get("t", np.full(row.shape, np.nan)), dtype=np.float64)
        p = np.asarray(data.get("p", np.full(row.shape, np.nan)), dtype=np.float64)
    finite = np.isfinite(row) & np.isfinite(col)
    row = row[finite].astype(np.int64)
    col = col[finite].astype(np.int64)
    t = t[finite] if t.shape[0] == finite.shape[0] else np.full(row.shape, np.nan)
    p = p[finite] if p.shape[0] == finite.shape[0] else np.full(row.shape, np.nan)
    valid = (row >= 0) & (col >= 0)
    return row[valid], col[valid], t[valid], p[valid]


def accumulation_image(row, col):
    if row.size == 0 or col.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    height = int(row.max()) + 1
    width = int(col.max()) + 1
    image = np.zeros((height, width), dtype=np.float32)
    np.add.at(image, (row, col), 1)
    return image


def save_image(path, image, log=False):
    if plt is None:
        return
    data = np.log1p(image) if log else image
    plt.figure(figsize=(8, 5))
    plt.imshow(data, cmap="magma", origin="upper", aspect="auto")
    plt.colorbar(label="log1p(events)" if log else "events")
    plt.xlabel("col")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_projection(path, values, axis_name):
    if plt is None:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.xlabel(axis_name)
    plt.ylabel("events")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def bbox_from_mask(mask, margin, width, height):
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return ""
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(width - 1, int(xs.max()) + margin)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(height - 1, int(ys.max()) + margin)
    return f"{x0},{y0},{x1 - x0 + 1},{y1 - y0 + 1}"


def event_bbox(image, margin):
    height, width = image.shape
    return bbox_from_mask(image > 0, margin, width, height)


def strongest_roi(image, margin):
    height, width = image.shape
    if float(image.sum()) <= 0:
        return ""
    threshold = max(1.0, float(np.percentile(image[image > 0], 90)) if np.any(image > 0) else 1.0)
    mask = image >= threshold
    return bbox_from_mask(mask, margin, width, height)


def hot_pixel_mask(image):
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return np.zeros_like(image, dtype=bool), float("nan")
    median = float(np.median(nonzero))
    mad = float(np.median(np.abs(nonzero - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(median + 10.0 * robust_sigma, float(np.percentile(nonzero, 99.9)), 50.0)
    return image > threshold, threshold


def block_stats(values, blocks):
    total = float(values.sum())
    splits = np.array_split(values, blocks)
    stats = {}
    for idx, block in enumerate(splits):
        count = float(block.sum())
        stats[f"block_{idx}_events"] = count
        stats[f"block_{idx}_fraction"] = count / (total + EPS)
    return stats


def quadrant_stats(image):
    h, w = image.shape
    row_mid = h // 2
    col_mid = w // 2
    return {
        "q_top_left_events": float(image[:row_mid, :col_mid].sum()),
        "q_top_right_events": float(image[:row_mid, col_mid:].sum()),
        "q_bottom_left_events": float(image[row_mid:, :col_mid].sum()),
        "q_bottom_right_events": float(image[row_mid:, col_mid:].sum()),
    }


def analyze_file(csv_path, out_dir, max_events, col_blocks, row_blocks, margin):
    row_coord, col_coord, t, p = read_events(csv_path, max_events)
    meta = parse_metadata(csv_path)
    image = accumulation_image(row_coord, col_coord)
    height, width = image.shape
    col_proj = image.sum(axis=0)
    row_proj = image.sum(axis=1)
    total = float(image.sum())
    col_mid = width // 2
    row_mid = height // 2
    left_events = float(image[:, :col_mid].sum())
    right_events = float(image[:, col_mid:].sum())
    top_events = float(image[:row_mid, :].sum())
    bottom_events = float(image[row_mid:, :].sum())

    hot_mask, hot_threshold = hot_pixel_mask(image)
    clean_image = image.copy()
    clean_image[hot_mask] = 0
    row_min = int(row_coord.min()) if row_coord.size else -1
    row_max = int(row_coord.max()) if row_coord.size else -1
    col_min = int(col_coord.min()) if col_coord.size else -1
    col_max = int(col_coord.max()) if col_coord.size else -1

    file_dir = Path(out_dir) / safe_file_output_name(csv_path)
    ensure_dir(file_dir)
    save_image(file_dir / "accumulation_linear.png", image, log=False)
    save_image(file_dir / "accumulation_log.png", image, log=True)
    save_projection(file_dir / "col_projection.png", col_proj, "col")
    save_projection(file_dir / "row_projection.png", row_proj, "row")
    save_projection(file_dir / "x_projection.png", col_proj, "col")
    save_projection(file_dir / "y_projection.png", row_proj, "row")

    row = {
        "file_path": str(csv_path),
        "file_name": Path(csv_path).name,
        "output_dir": str(file_dir),
        "threshold": meta["threshold"],
        "phantom": meta["phantom"],
        "velocity": meta["velocity"],
        "events_loaded": int(row_coord.size),
        "image_width": int(width),
        "image_height": int(height),
        "col_min": col_min,
        "col_max": col_max,
        "row_min": row_min,
        "row_max": row_max,
        "total_events": total,
        "col_left_events": left_events,
        "col_right_events": right_events,
        "left_right_event_ratio": left_events / (right_events + EPS),
        "top_events": top_events,
        "bottom_events": bottom_events,
        "top_bottom_event_ratio": top_events / (bottom_events + EPS),
        "hot_pixel_count": int(hot_mask.sum()),
        "hot_pixel_threshold": hot_threshold,
        "recommended_left_roi": f"0,{max(row_min, 0)},{col_mid},{max(row_max - row_min + 1, 0)}" if row_min >= 0 else "",
        "recommended_right_roi": f"{col_mid},{max(row_min, 0)},{width - col_mid},{max(row_max - row_min + 1, 0)}" if row_min >= 0 else "",
        "recommended_strongest_roi": strongest_roi(image, margin),
        "recommended_hot_pixel_removed_roi": event_bbox(clean_image, margin),
        "recommended_col_range": f"{col_min}-{col_max}" if col_min >= 0 else "",
        "recommended_row_range": f"{row_min}-{row_max}" if row_min >= 0 else "",
    }
    row.update(quadrant_stats(image))
    for key, value in block_stats(col_proj, col_blocks).items():
        row[f"col_{key}"] = value
    for key, value in block_stats(row_proj, row_blocks).items():
        row[f"row_{key}"] = value
    return row


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "file_name",
        "threshold",
        "phantom",
        "velocity",
        "events_loaded",
        "image_width",
        "image_height",
        "col_min",
        "col_max",
        "row_min",
        "row_max",
        "col_left_events",
        "col_right_events",
        "left_right_event_ratio",
        "recommended_col_range",
        "recommended_row_range",
        "hot_pixel_count",
        "recommended_left_roi",
        "recommended_right_roi",
        "recommended_strongest_roi",
        "recommended_hot_pixel_removed_roi",
        "file_path",
        "output_dir",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(headers, rows, max_rows=None):
    selected = rows[:max_rows] if max_rows else rows
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selected:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    if not selected:
        lines.append("| " + " | ".join(["-"] * len(headers)) + " |")
    return "\n".join(lines)


def write_report(path, rows, args):
    lines = [
        "# Event Accumulation ROI Diagnostic",
        "",
        "This script only analyzes event CSV files. It does not train models or modify datasets.",
        "",
        "## Config",
        "",
        f"- roots: `{args.roots}`",
        f"- out_dir: `{args.out_dir}`",
        f"- max_events: `{args.max_events}`",
        f"- col_blocks: `{args.col_blocks}`",
        f"- row_blocks: `{args.row_blocks}`",
        "",
        "## Summary",
        "",
        markdown_table(
            [
                "file_name",
                "threshold",
                "phantom",
                "velocity",
                "events_loaded",
                "row_min",
                "row_max",
                "col_left_events",
                "col_right_events",
                "left_right_event_ratio",
                "recommended_col_range",
                "recommended_row_range",
                "hot_pixel_count",
            ],
            rows,
        ),
        "",
        "## Recommended ROIs",
        "",
        markdown_table(
            [
                "file_name",
                "recommended_left_roi",
                "recommended_right_roi",
                "recommended_strongest_roi",
                "recommended_hot_pixel_removed_roi",
            ],
            rows,
        ),
        "",
        "## Celex ROI Horizontal Band",
        "",
        "Use `row_min/row_max` to confirm which horizontal band the current Celex ROI actually covers.",
        "",
    ]
    if rows:
        valid_rows = [row for row in rows if row.get("row_min", -1) >= 0 and row.get("row_max", -1) >= 0]
        if valid_rows:
            row_min = min(row["row_min"] for row in valid_rows)
            row_max = max(row["row_max"] for row in valid_rows)
            lines.append(f"- Overall row_min/row_max across files: `{row_min}-{row_max}`")
        else:
            lines.append("- Overall row_min/row_max across files: `N/A`")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    if np is None:
        raise RuntimeError(
            "NumPy is required for this diagnostic script. Install numpy in the Python environment used to run it. "
            f"Original import error: {NP_IMPORT_ERROR}"
        )
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    csv_files = find_csv_files(args.roots)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under roots: {args.roots}")

    rows = []
    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Analyzing {csv_path}")
        try:
            rows.append(analyze_file(csv_path, out_dir, args.max_events, args.col_blocks, args.row_blocks, args.margin))
        except Exception as exc:
            print(f"Warning: failed to analyze {csv_path}: {exc}")
            meta = parse_metadata(csv_path)
            rows.append(
                {
                    "file_path": str(csv_path),
                    "file_name": Path(csv_path).name,
                    "threshold": meta["threshold"],
                    "phantom": meta["phantom"],
                    "velocity": meta["velocity"],
                    "error": str(exc),
                }
            )

    write_csv(out_dir / "roi_summary.csv", rows)
    write_report(out_dir / "roi_report.md", rows, args)
    print(f"Saved summary: {out_dir / 'roi_summary.csv'}")
    print(f"Saved report: {out_dir / 'roi_report.md'}")


if __name__ == "__main__":
    main()
