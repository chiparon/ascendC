#!/usr/bin/env python3
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ASYNC_DIR = ROOT / "MatmulLeakyReluInvocationAsync"
LEGACY_BASELINE_DIR = ROOT.parent.parent / "MatmulLeakyReluInvocationAsync"

SHAPES = [
    (2048, 2048, 2048),
    (4096, 1024, 4096),
]
REPEAT = int(os.getenv("MATMUL_BENCH_REPEAT", "5"))
RUN_MODE = os.getenv("MATMUL_BENCH_MODE", "cpu")
SOC = os.getenv("MATMUL_BENCH_SOC", "Ascend910B1")

AVG_RE = re.compile(r"\[PERF\] AVG_MS=([0-9.]+)")
P50_RE = re.compile(r"\[PERF\] P50_MS=([0-9.]+)")
P90_RE = re.compile(r"\[PERF\] P90_MS=([0-9.]+)")
ERR_RE = re.compile(r"error ratio:\s*([0-9.]+)")


def parse_metrics(text: str):
    def _find(regex):
        m = regex.search(text)
        if not m:
            raise RuntimeError(f"missing metric for regex: {regex.pattern}")
        return float(m.group(1))

    return {
        "avg_ms": _find(AVG_RE),
        "p50_ms": _find(P50_RE),
        "p90_ms": _find(P90_RE),
        "error_ratio": _find(ERR_RE),
        "test_pass": "test pass" in text,
    }


def extra_error_metrics(case_dir: Path):
    out = np.fromfile(case_dir / "output/output.bin", dtype=np.float32)
    golden = np.fromfile(case_dir / "output/golden.bin", dtype=np.float32)
    if out.size != golden.size:
        raise RuntimeError(f"size mismatch: out={out.size}, golden={golden.size}")
    diff = np.abs(out - golden)
    return {
        "max_abs_err": float(diff.max(initial=0.0)),
        "mean_abs_err": float(diff.mean() if diff.size else 0.0),
    }


def run_async_variant(shape, force_core):
    m, n, k = shape
    cmd = [
        "bash",
        "run.sh",
        "-r",
        RUN_MODE,
        "-v",
        SOC,
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--repeat",
        str(REPEAT),
        "--force-core",
        str(force_core),
    ]
    proc = subprocess.run(cmd, cwd=ASYNC_DIR, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"run failed (shape={shape}, force_core={force_core})\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    metrics = parse_metrics(proc.stdout)
    metrics.update(extra_error_metrics(ASYNC_DIR))
    metrics["shape"] = [m, n, k]
    metrics["force_core"] = force_core
    return metrics


def run_legacy_baseline_once():
    cmd = ["bash", "run.sh", "-r", RUN_MODE, "-v", SOC]
    proc = subprocess.run(cmd, cwd=LEGACY_BASELINE_DIR, text=True, capture_output=True)
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-800:],
    }


def main():
    results = {
        "meta": {
            "repeat": REPEAT,
            "run_mode": RUN_MODE,
            "soc": SOC,
            "async_dir": str(ASYNC_DIR),
            "legacy_baseline_dir": str(LEGACY_BASELINE_DIR),
        },
        "legacy_baseline_probe": run_legacy_baseline_once(),
        "rows": [],
    }

    for shape in SHAPES:
        baseline = run_async_variant(shape, force_core=1)
        optimized = run_async_variant(shape, force_core=0)
        speedup = (baseline["avg_ms"] - optimized["avg_ms"]) / baseline["avg_ms"] * 100.0
        results["rows"].append(
            {
                "shape": list(shape),
                "baseline": baseline,
                "optimized": optimized,
                "speedup_pct": speedup,
            }
        )

    report_path = ROOT / "benchmark_async_compare.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[INFO] wrote {report_path}")
    for row in results["rows"]:
        shape = tuple(row["shape"])
        print(
            "[RESULT] shape={} baseline(avg/p50/p90)={:.3f}/{:.3f}/{:.3f} ms, "
            "optimized={:.3f}/{:.3f}/{:.3f} ms, speedup={:.2f}%, "
            "max_abs_err(opt)={:.6e}".format(
                shape,
                row["baseline"]["avg_ms"],
                row["baseline"]["p50_ms"],
                row["baseline"]["p90_ms"],
                row["optimized"]["avg_ms"],
                row["optimized"]["p50_ms"],
                row["optimized"]["p90_ms"],
                row["speedup_pct"],
                row["optimized"]["max_abs_err"],
            )
        )


if __name__ == "__main__":
    main()
