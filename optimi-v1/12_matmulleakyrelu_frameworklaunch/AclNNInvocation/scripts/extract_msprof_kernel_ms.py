#!/usr/bin/env python3
"""Best-effort parser for msprof op output.

This script scans csv files under one msprof output root, extracts per-run
kernel time for rows that match a keyword pattern, and reports AVG/P50/P90.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable


def _to_number(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    cleaned = re.sub(r"[^0-9eE+\-\.]", "", cleaned)
    if cleaned in ("", "+", "-", ".", "+.", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _unit_scale_to_ms(header: str) -> float:
    low = header.lower()
    if "(ns" in low or " ns" in low:
        return 1e-6
    if "(us" in low or " us" in low or "μs" in low or "µs" in low:
        return 1e-3
    if "(ms" in low or " ms" in low:
        return 1.0
    if "(s" in low or " sec" in low:
        return 1000.0
    return 1.0


def _header_priority(header: str) -> int:
    token = re.sub(r"[^a-z0-9]", "", header.lower())
    if "taskduration" in token:
        return 0
    if "totaltime" in token:
        return 1
    if "duration" in token:
        return 2
    if "time" in token:
        return 3
    return 99


def _pick_time_columns(headers: Iterable[str]) -> list[tuple[int, int, float]]:
    picked: list[tuple[int, int, float]] = []
    for idx, header in enumerate(headers):
        priority = _header_priority(header)
        if priority >= 99:
            continue
        picked.append((priority, idx, _unit_scale_to_ms(header)))
    picked.sort(key=lambda x: x[0])
    return picked


def _list_run_dirs(msprof_root: Path) -> list[Path]:
    run_dirs = sorted(
        p for p in msprof_root.glob("run_*") if p.is_dir() and p.name[4:].isdigit()
    )
    if run_dirs:
        return run_dirs
    return [msprof_root]


def _score_file(path: Path) -> int:
    low = path.name.lower()
    if "op_summary" in low:
        return 0
    if "op_statistic" in low or "op_stat" in low:
        return 1
    return 2


def _extract_run_ms(run_dir: Path, keyword: re.Pattern[str]) -> tuple[float | None, str]:
    csv_files = sorted(run_dir.rglob("*.csv"))
    best_value: float | None = None
    best_source = ""
    best_rank: tuple[int, int, float] | None = None

    for csv_path in csv_files:
        rank_file = _score_file(csv_path)
        try:
            with csv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                time_cols = _pick_time_columns(header)
                if not time_cols:
                    continue

                # pass 1: rows matched by keyword
                totals = [0.0 for _ in time_cols]
                matched_rows = 0
                for row in reader:
                    line = " ".join(row).lower()
                    if keyword.search(line) is None:
                        continue
                    matched_rows += 1
                    for i, (_, col_idx, scale) in enumerate(time_cols):
                        if col_idx >= len(row):
                            continue
                        num = _to_number(row[col_idx])
                        if num is None:
                            continue
                        totals[i] += num * scale

                # fallback: if no keyword matched, use all rows for op_summary/op_stat files
                use_fallback = matched_rows == 0 and rank_file <= 1
                if use_fallback:
                    f.seek(0)
                    reader = csv.reader(f)
                    next(reader, None)
                    totals = [0.0 for _ in time_cols]
                    for row in reader:
                        for i, (_, col_idx, scale) in enumerate(time_cols):
                            if col_idx >= len(row):
                                continue
                            num = _to_number(row[col_idx])
                            if num is None:
                                continue
                            totals[i] += num * scale
                    matched_rows = -1  # marker for fallback mode

                selected = None
                selected_col_pri = 99
                for i, (priority, _, _) in enumerate(time_cols):
                    if totals[i] <= 0:
                        continue
                    if priority < selected_col_pri:
                        selected = totals[i]
                        selected_col_pri = priority

                if selected is None:
                    continue

                row_rank = 1 if matched_rows > 0 else (2 if matched_rows == -1 else 3)
                rank = (rank_file, row_rank, float(selected_col_pri))
                if best_rank is None or rank < best_rank or (
                    rank == best_rank and (best_value is None or selected > best_value)
                ):
                    best_rank = rank
                    best_value = selected
                    best_source = str(csv_path)
        except OSError:
            continue

    return best_value, best_source


def _percentile(values: list[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msprof-root", required=True, help="msprof output root")
    parser.add_argument(
        "--pattern",
        default=r"matmul|leaky|custom",
        help="case-insensitive regex used to match target op rows",
    )
    args = parser.parse_args()

    msprof_root = Path(args.msprof_root)
    if not msprof_root.exists():
        print("[KERNEL] AVG_MS=NA")
        print("[KERNEL] P50_MS=NA")
        print("[KERNEL] P90_MS=NA")
        print("[KERNEL] SAMPLES=0")
        print("[KERNEL] SOURCE=NA")
        return 0

    try:
        keyword = re.compile(args.pattern, re.IGNORECASE)
    except re.error:
        keyword = re.compile(r"matmul|leaky|custom", re.IGNORECASE)

    run_values: list[float] = []
    run_sources: list[str] = []
    for run_dir in _list_run_dirs(msprof_root):
        value, source = _extract_run_ms(run_dir, keyword)
        if value is None:
            continue
        run_values.append(value)
        run_sources.append(source)

    if not run_values:
        print("[KERNEL] AVG_MS=NA")
        print("[KERNEL] P50_MS=NA")
        print("[KERNEL] P90_MS=NA")
        print("[KERNEL] SAMPLES=0")
        print("[KERNEL] SOURCE=NA")
        return 0

    run_values_sorted = sorted(run_values)
    avg = sum(run_values_sorted) / len(run_values_sorted)
    p50 = _percentile(run_values_sorted, 0.50)
    p90 = _percentile(run_values_sorted, 0.90)

    print(f"[KERNEL] AVG_MS={avg:.3f}")
    print(f"[KERNEL] P50_MS={p50:.3f}")
    print(f"[KERNEL] P90_MS={p90:.3f}")
    print(f"[KERNEL] SAMPLES={len(run_values_sorted)}")
    print(f"[KERNEL] SOURCE={run_sources[-1] if run_sources else 'NA'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
