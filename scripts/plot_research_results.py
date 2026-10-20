#!/usr/bin/env python3
"""Rebuild research figures from repository evidence; no model weights or fitting."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research-2026-10-20"
QUALITY = Path("results/quality-2026-10-20")
COLORS = {
    "H100": "#2563a6",
    "H200": "#058b72",
    "GH200": "#8661ac",
    "Spark GB10": "#df7833",
    "H200 new session": "#087f96",
}
WORKLOADS = ["vector_add", "memcpy_bandwidth", "stencil_1d", "reduction", "gemm_naive", "gemm_tiled"]
WC = dict(zip(WORKLOADS, ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#6b7280", "#E69F00"], strict=True))
SHORT = {
    "vector_add": "Vector add",
    "memcpy_bandwidth": "Memcpy",
    "stencil_1d": "Stencil",
    "reduction": "Reduction",
    "gemm_naive": "GEMM naive",
    "gemm_tiled": "GEMM tiled",
}
INPUTS: dict[str, str] = {}
CAPTIONS: dict[str, str] = {}
CHECKS: dict[str, object] = {}


def register(path: Path) -> Path:
    if path.parts[0] != "results" or "outputs" in path.parts:
        raise ValueError("Figures require repository results inputs")
    absolute = ROOT / path
    INPUTS[str(path)] = hashlib.sha256(absolute.read_bytes()).hexdigest()
    return absolute


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(register(path))


def read_json(path: Path) -> dict:
    return json.loads(register(path).read_text())


def data(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT / "data" / (name + ".csv"), index=False)


def save(fig, name: str, caption: str, footnote: str, bottom: float = 0.22) -> None:
    fig.text(0.015, 0.012, footnote, fontsize=8.3, color="#475569", va="bottom")
    fig.subplots_adjust(bottom=bottom, top=0.82, wspace=0.34, hspace=0.6)
    for extension in ["png", "svg"]:
        fig.savefig(OUT / f"{name}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CAPTIONS[name] = caption


def capture_paths() -> dict[str, Path]:
    return {
        "H100": Path("results/h100-power-2026-09-22"),
        "H200": QUALITY / "h200/power",
        "GH200": QUALITY / "gh200/power",
        "Spark GB10": QUALITY / "spark/power",
        "H200 new session": QUALITY / "h200-independent/power",
    }


def coverage() -> None:
    rows = []
    for label, directory in capture_paths().items():
        manifest = read_json(directory / "manifest.json")
        raw, steady, windows = (
            csv(directory / "telemetry.csv"),
            csv(directory / "steady_telemetry.csv"),
            csv(directory / "summary.csv"),
        )
        assert len(raw) == manifest["telemetry_samples"] and len(steady) == manifest["steady_samples"]
        assert len(windows) == 24 and windows["trial"].nunique() == 3
        rows.append(
            {
                "capture": label,
                "raw_rows": len(raw),
                "steady_rows": len(steady),
                "workload_windows": len(windows),
                "trials": 3,
                "role": "Development"
                if label in {"H100", "H200"}
                else "Prospective known GPU"
                if label == "H200 new session"
                else "External device",
            }
        )
    frame = pd.DataFrame(rows)
    data("capture_coverage", frame)
    CHECKS["capture_counts"] = rows
    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    x = np.arange(len(rows))
    for offset, field, color, label in [
        (-0.18, "raw_rows", "#cbd5e1", "All sensor rows"),
        (0.18, "steady_rows", "#2563a6", "Retained steady rows"),
    ]:
        bars = ax.bar(x + offset, frame[field], width=0.34, color=color, label=label)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(
        x,
        [f"{row['capture']}\n{row['role']}" for row in rows],
        fontsize=9,
    )
    ax.set_ylabel("Telemetry observations (rows)")
    ax.set_ylim(0, max(frame["raw_rows"]) * 1.2)
    ax.legend(frameon=False, ncol=2)
    fig.suptitle("Five measured captures; two development devices", x=0.015, ha="left", fontsize=16, weight="bold")
    ax.set_title(
        "Every capture: 8 configurations × 3 randomized trials = 24 verified workload windows", loc="left", fontsize=10
    )
    save(
        fig,
        "01_capture_coverage",
        "Captured observations and retained steady rows, with development and external/prospective roles explicitly separated. Each capture contains 24 verified workload windows across three trials.",
        "Rows are temporally correlated; observation counts are not independent experimental replicates. A100 collection remains queued and is excluded.",
    )


def timing() -> None:
    paths = {
        "H100": Path("results/h100-2026-09-22"),
        "H200": QUALITY / "h200/timing",
        "GH200": QUALITY / "gh200/timing",
        "Spark GB10": QUALITY / "spark/timing",
    }
    keys = ["kernel", "problem_size", "block_size"]
    rows = []
    for trial in [1, 2, 3]:
        baseline = csv(paths["H100"] / f"trial-{trial}/benchmark_summary.csv")
        assert len(baseline) == 74 and not baseline.duplicated(keys).any()
        for device in ["H200", "GH200", "Spark GB10"]:
            candidate = csv(paths[device] / f"trial-{trial}/benchmark_summary.csv")
            joined = baseline[keys + ["runtime_ms_mean"]].merge(
                candidate[keys + ["runtime_ms_mean"]], on=keys, validate="one_to_one", suffixes=("_h100", "_candidate")
            )
            assert len(joined) == len(candidate) == 74
            joined["ratio"] = joined["runtime_ms_mean_candidate"] / joined["runtime_ms_mean_h100"]
            joined["device"], joined["trial"] = device, trial
            rows.append(joined)
    all_rows = pd.concat(rows, ignore_index=True)
    data("matched_timing_trials", all_rows)
    grouped = all_rows.groupby(["device"] + keys)["ratio"].agg(["median", "min", "max"]).reset_index()
    data("matched_timing_ranges", grouped)
    CHECKS["timing_ratio_full_range"] = [float(all_rows["ratio"].min()), float(all_rows["ratio"].max())]
    CHECKS["matched_timing_scenarios_per_device_trial"] = 74
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=True)
    for ax, device in zip(axes, ["H200", "GH200", "Spark GB10"], strict=True):
        frame = grouped[grouped["device"] == device]
        for index, workload in enumerate(WORKLOADS):
            group = frame[frame["kernel"] == workload].sort_values(["problem_size", "block_size"])
            positions = index + np.linspace(-0.30, 0.30, len(group))
            ax.vlines(positions, group["min"], group["max"], colors=COLORS[device], alpha=0.35, lw=1)
            ax.scatter(positions, group["median"], color=COLORS[device], s=16, zorder=3)
        ax.axhline(1, color="#475569", lw=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_xticks(range(len(WORKLOADS)), [SHORT[w] for w in WORKLOADS], rotation=55, ha="right", fontsize=8)
        ax.set_title(device, color=COLORS[device], weight="bold")
        ax.set_ylim(float(all_rows["ratio"].min()) * 0.88, float(all_rows["ratio"].max()) * 1.15)
        ax.grid(axis="y", alpha=0.2, which="both")
    axes[0].set_ylabel("Mean runtime / corresponding H100 runtime\n(log scale; lower is faster)")
    fig.suptitle(
        "All 74 matched timing scenarios, with three-trial variation", x=0.015, ha="left", fontsize=16, weight="bold"
    )
    save(
        fig,
        "02_matched_timing",
        "Each dot is one scenario's median runtime ratio across three corresponding trial pairs; vertical segments span those three ratios. All 74 scenario identities match for every device and trial. The dashed line is H100 parity.",
        "Ranges are descriptive, not confidence intervals. Devices, host CPU, toolkit and capture dates differ; cross-device CI gates are ineligible.",
    )


def power() -> None:
    rows = []
    for device, directory in list(capture_paths().items())[:4]:
        frame = csv(directory / "summary.csv")
        frame["device"] = device
        frame["sustained_ms_per_launch"] = 1000 / frame["launches_per_second"]
        np.testing.assert_allclose(
            frame["estimated_j_per_launch"], frame["mean_power_w"] / frame["launches_per_second"], rtol=1e-10
        )
        rows.append(frame)
    trials = pd.concat(rows, ignore_index=True)
    data("power_tradeoff_trials", trials)
    fields = ["sustained_ms_per_launch", "estimated_j_per_launch", "mean_power_w"]
    grouped = trials.groupby(["device", "kernel", "problem_size", "block_size"])[fields].agg(["median", "min", "max"])
    data(
        "power_tradeoff_ranges",
        grouped.reset_index().set_axis(
            ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in grouped.reset_index().columns], axis=1
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    markers = {"H100": "o", "H200": "s", "GH200": "^", "Spark GB10": "D"}
    for (device, workload, _, _), row in grouped.iterrows():
        x = row[("sustained_ms_per_launch", "median")]
        for ax, field in zip(axes, ["estimated_j_per_launch", "mean_power_w"], strict=True):
            y = row[(field, "median")]
            ax.errorbar(
                x,
                y,
                xerr=[[x - row[("sustained_ms_per_launch", "min")]], [row[("sustained_ms_per_launch", "max")] - x]],
                yerr=[[y - row[(field, "min")]], [row[(field, "max")] - y]],
                fmt=markers[device],
                ms=6,
                color=WC[workload],
                alpha=0.82,
                lw=0.8,
                capsize=2,
            )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Sustained elapsed ms / launch (log scale)")
        ax.grid(alpha=0.18, which="both")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Estimated joules / launch (log scale)")
    axes[1].set_ylabel("Measured GPU sensor power (W)")
    axes[0].set_title("Energy–runtime tradeoff")
    axes[1].set_title("Power–runtime tradeoff")
    fig.legend(
        handles=[Line2D([], [], color=WC[w], marker="o", linestyle="", label=SHORT[w]) for w in WORKLOADS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    fig.legend(
        handles=[Line2D([], [], color="#475569", marker=m, linestyle="", label=d) for d, m in markers.items()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.145),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Eight sustained configurations on four devices", x=0.015, ha="left", fontsize=16, weight="bold")
    save(
        fig,
        "03_power_energy_tradeoffs",
        "Each point represents a device/configuration; points and ranges are the median and full range of three trial summaries. Estimated joules per launch equal steady mean sensor watts divided by whole-window launch rate. Runtime includes host launch and synchronization overhead.",
        "Spark sensor measures GPU-only power; discrete-device captures use GPU-board telemetry. Neither is whole-system power. Ranges are not confidence intervals.",
        bottom=0.32,
    )


def selection() -> None:
    frame = csv(QUALITY / "models/model-development/fold_metrics.csv")
    device_selection = read_json(QUALITY / "models/model-development/selection.json")
    known_selection = read_json(QUALITY / "models/known-device-development/selection.json")
    assert device_selection["candidate"]["id"] == "training_workload_mean"
    assert known_selection["candidate"]["id"] == "rf_unbounded_full"
    np.testing.assert_allclose(
        frame.query("split=='device' and candidate=='training_workload_mean'")["rmse_w"].mean(),
        device_selection["mean_device_rmse_w"],
    )
    np.testing.assert_allclose(
        frame.query("split=='whole_trial_diagnostic' and candidate=='rf_unbounded_full'")["rmse_w"].mean(),
        known_selection["mean_trial_rmse_w"],
    )
    names = {"training_mean": "Global mean", "training_workload_mean": "Workload mean"}
    for candidate in frame["candidate"].unique():
        if candidate not in names:
            name = (
                candidate.replace("_portable", " · portable")
                .replace("_full", " · full")
                .replace("rf_unbounded", "RF64 unrestricted")
                .replace("rf_depth6", "RF64 depth 6")
                .replace("gb_lr005", "GB64 lr .05")
                .replace("gb_lr01", "GB64 lr .10")
                .replace("linear", "Linear")
            )
            names[candidate] = name
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7))
    summary = []
    for ax, split, title, winner in zip(
        axes,
        ["device", "whole_trial_diagnostic"],
        ["Cross-device objective · 2 device folds", "Known-device objective · 6 whole-trial folds"],
        ["training_workload_mean", "rf_unbounded_full"],
        strict=True,
    ):
        folds = frame[frame["split"] == split]
        order = folds.groupby("candidate")["rmse_w"].mean().sort_values().index
        for index, candidate in enumerate(order):
            values = folds[folds["candidate"] == candidate]["rmse_w"].to_numpy()
            color = "#058b72" if candidate == winner else "#64748b" if candidate.startswith("training_") else "#a7bdd1"
            ax.barh(index, values.mean(), color=color, height=0.65)
            ax.scatter(values, np.full(len(values), index), color="#1e293b", s=11, zorder=3)
            summary.append(
                {
                    "objective": split,
                    "candidate": candidate,
                    "mean_rmse_w": values.mean(),
                    "min_rmse_w": values.min(),
                    "max_rmse_w": values.max(),
                    "folds": len(values),
                    "selected": candidate == winner,
                }
            )
        ax.set_yticks(range(len(order)), [names[name] for name in order], fontsize=8.3)
        ax.invert_yaxis()
        ax.set_xlabel("Validation RMSE (W); lower is better")
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(axis="x", alpha=0.2)
    data("model_selection_summary", pd.DataFrame(summary))
    CHECKS["cross_device_winner_mean_rmse_w"] = float(
        frame.query("split=='device' and candidate=='training_workload_mean'")["rmse_w"].mean()
    )
    CHECKS["known_device_winner_mean_rmse_w"] = float(
        frame.query("split=='whole_trial_diagnostic' and candidate=='rf_unbounded_full'")["rmse_w"].mean()
    )
    fig.suptitle(
        "Different validation objectives select different models", x=0.015, ha="left", fontsize=16, weight="bold"
    )
    save(
        fig,
        "04_model_selection",
        "Bars are unweighted mean fold RMSE; dots show every held-out fold. A workload mean wins the two-device objective. A forest wins the separate known-device objective. Both selections were frozen before their subsequent external/prospective tests.",
        "These are development/selection scores, not final-test accuracy. Axes have separate ranges; whole-trial validation retains other trials from the same GPU.",
    )


def prospective() -> None:
    directory = QUALITY / "h200-independent/evaluation"
    frame = csv(directory / "predictions.csv")
    evidence = read_json(directory / "evaluation.json")
    actual = frame["power_watts"].to_numpy()
    predicted = frame["rf_unbounded_full_predicted_w"].to_numpy()
    frame["signed_error_w"] = predicted - actual
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    np.testing.assert_allclose(rmse, evidence["scores"]["rf_unbounded_full"]["rmse_w"], rtol=1e-12)
    tail = int((np.abs(frame["signed_error_w"]) > 25).sum())
    CHECKS["prospective_h200"] = {
        "rows": len(frame),
        "rmse_w": rmse,
        "errors_over_25w": tail,
        "max_absolute_error_w": float(np.max(np.abs(predicted - actual))),
    }
    data("prospective_h200_errors", frame)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5))
    for workload in WORKLOADS:
        rows = frame[frame["workload_type"] == workload]
        axes[0].scatter(
            rows["power_watts"],
            rows["rf_unbounded_full_predicted_w"],
            s=11,
            alpha=0.45,
            color=WC[workload],
            label=SHORT[workload],
        )
    low, high = min(actual.min(), predicted.min()) - 5, max(actual.max(), predicted.max()) + 5
    axes[0].plot([low, high], [low, high], color="#475569", linestyle="--", lw=1)
    axes[0].set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel="Observed sensor power (W)",
        ylabel="Predicted sensor power (W)",
        title="Calibration · all 1,153 rows",
    )
    axes[1].hist(frame["signed_error_w"], bins=55, color="#2563a6")
    axes[1].axvline(0, color="#475569", linestyle="--", lw=1)
    axes[1].set(xlabel="Prediction − observation (W)", ylabel="Rows", title=f"Full residual tail · {tail} errors >25 W")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Rows (log scale)")
    groups = [frame.loc[frame["workload_type"] == w, "signed_error_w"] for w in WORKLOADS]
    boxes = axes[2].boxplot(
        groups,
        orientation="horizontal",
        patch_artist=True,
        tick_labels=[SHORT[w] for w in WORKLOADS],
        whis=(0, 100),
        showfliers=False,
    )
    for box, workload in zip(boxes["boxes"], WORKLOADS, strict=True):
        box.set_facecolor(WC[workload])
        box.set_alpha(0.6)
    axes[2].axvline(0, color="#475569", linestyle="--", lw=1)
    axes[2].set(xlabel="Prediction − observation (W)", title="Per-workload errors · full min/max")
    fig.legend(
        handles=[Patch(facecolor=WC[w], label=SHORT[w]) for w in WORKLOADS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"Prospective H200 session: {rmse:.2f} W RMSE, with an exposed error tail",
        x=0.015,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    save(
        fig,
        "05_prospective_h200",
        "A frozen forest evaluated once on a later H200 capture: calibration, a logarithmic-count residual histogram, and per-workload signed-error distributions. All 1,153 rows are retained, including 24 errors above 25 W. Box whiskers span the full observed minimum/maximum; boxes span quartiles.",
        "Independent collection session on a known GPU; not an unseen-device test. No clipping, residual trimming, target adaptation or model refit.",
    )


def external() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    rows = []
    for ax, device, title in zip(
        axes,
        ["gh200", "spark"],
        ["GH200 · discrete GPU-board sensor", "Spark GB10 · integrated GPU-only sensor"],
        strict=True,
    ):
        fixed, forest, historical = [
            read_json(QUALITY / device / name / "evaluation.json")
            for name in ["frozen-evaluation", "known-device-challenge", "transfer"]
        ]
        models = [
            ("Frozen workload mean", fixed["scores"]["training_workload_mean"]),
            ("Known-device forest", forest["scores"]["rf_unbounded_full"]),
            ("Historical H100 linear", historical["scores"]["linear_regression"]),
            ("Pooled global mean", fixed["scores"]["pooled_training_mean"]),
        ]
        checks = [
            ("frozen-evaluation", "training_workload_mean", fixed),
            ("known-device-challenge", "rf_unbounded_full", forest),
            ("transfer", "linear_regression", historical),
        ]
        for directory, model_name, evidence in checks:
            predictions = csv(QUALITY / device / directory / "predictions.csv")
            actual = predictions["power_watts"].to_numpy()
            values = predictions[model_name + "_predicted_w"].to_numpy()
            assert len(actual) == evidence["test_rows"]
            recomputed = float(np.sqrt(np.mean((values - actual) ** 2)))
            np.testing.assert_allclose(recomputed, evidence["scores"][model_name]["rmse_w"], rtol=1e-12)
            CHECKS[device + "_" + model_name + "_rmse_w"] = recomputed
        baseline_rows = csv(QUALITY / device / "frozen-evaluation/predictions.csv")
        baseline_rmse = float(
            np.sqrt(np.mean((baseline_rows["pooled_training_mean_predicted_w"] - baseline_rows["power_watts"]) ** 2))
        )
        np.testing.assert_allclose(baseline_rmse, fixed["scores"]["pooled_training_mean"]["rmse_w"], rtol=1e-12)
        values = [metrics["rmse_w"] for _, metrics in models]
        bars = ax.barh(range(4), values, color=["#058b72", "#2563a6", "#8661ac", "#94a3b8"])
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_yticks(range(4), [name for name, _ in models], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.23)
        ax.set(xlabel="External capture RMSE (W); lower is better", title=title)
        ax.grid(axis="x", alpha=0.2)
        for name, metrics in models:
            rows.append({"device": device, "model": name, "test_rows": fixed["test_rows"], **metrics})
    data("external_device_metrics", pd.DataFrame(rows))
    fig.suptitle(
        "External-device failures remain visible; the frozen winner stays fixed",
        x=0.015,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    save(
        fig,
        "06_external_device_failure",
        "Final RMSE on GH200 and Spark with separate axis ranges. The historical fixed H100 linear reference is shown retrospectively, not promoted to a new winner. The known-device forest is challenged off device without refitting; its low within-device score does not transfer.",
        "Axes deliberately differ. Spark wattage is GPU-only, not whole-SoC/system or necessarily the same sensor scope as discrete boards. No post-test tuning.",
    )


def domains() -> None:
    frames = {device: csv(path / "steady_telemetry.csv") for device, path in capture_paths().items()}
    features = [
        ("memory_utilization_pct", "Memory-controller utilization (%)"),
        ("temperature_c", "GPU temperature (°C)"),
        ("graphics_clock_mhz", "Graphics clock (MHz)"),
        ("memory_clock_mhz", "Memory clock (MHz)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    rows = []
    for ax, (feature, label) in zip(axes.flat, features, strict=True):
        for position, (device, frame) in enumerate(frames.items(), 1):
            values = pd.to_numeric(frame[feature], errors="coerce").dropna()
            rows.append(
                {
                    "capture": device,
                    "feature": feature,
                    "available": len(values),
                    "missing": len(frame) - len(values),
                    "min": values.min() if len(values) else None,
                    "median": values.median() if len(values) else None,
                    "max": values.max() if len(values) else None,
                }
            )
            if len(values):
                box = ax.boxplot(
                    [values], positions=[position], widths=0.5, patch_artist=True, whis=(0, 100), showfliers=False
                )
                box["boxes"][0].set_facecolor(COLORS[device])
                box["boxes"][0].set_alpha(0.6)
            else:
                ax.text(
                    position,
                    0.5,
                    "N/A",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    color=COLORS[device],
                    weight="bold",
                )
        ax.set_xticks(
            range(1, 6),
            ["H100\ndev", "H200\ndev", "GH200\nexternal", "Spark\nexternal", "H200\nnew session"],
            fontsize=8,
        )
        ax.set_ylabel(label)
        ax.set_xlim(0.5, 5.5)
        ax.grid(axis="y", alpha=0.2)
    data("feature_domain_summary", pd.DataFrame(rows))
    fig.suptitle(
        "Feature domains differ across development and test captures", x=0.015, ha="left", fontsize=16, weight="bold"
    )
    save(
        fig,
        "07_feature_domains",
        "Observed telemetry feature distributions by capture. Boxes span quartiles; whiskers include the full minimum/maximum. Static memory clocks distinguish the development GPUs. Spark's unsupported memory-clock sensor remains missing, not zero.",
        "These are input distributions, not feature importances or causal effects. No model weights are needed or inferred; no importance estimate is fabricated.",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 12,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    for make in [coverage, timing, power, selection, prospective, external, domains]:
        make()
    manifest = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_sha256": INPUTS,
        "calculation_checks": CHECKS,
        "figures": {
            name: {
                "caption": caption,
                "files": {
                    extension: hashlib.sha256((OUT / f"{name}.{extension}").read_bytes()).hexdigest()
                    for extension in ["png", "svg"]
                },
            }
            for name, caption in CAPTIONS.items()
        },
        "data_sha256": {
            str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((OUT / "data").glob("*.csv"))
        },
    }

    lines = [
        "# Research figures — measured evidence and model limits",
        "",
        "Rebuild from repository inputs with `.venv/bin/python scripts/plot_research_results.py`. No files under `outputs/`, executable model bundles, model fitting, or network data are used. Matplotlib renders PNG and editable SVG with the Agg backend. The immutable quality evidence remains unchanged.",
        "",
        "`figure-manifest.json` records generator/environment versions, every input SHA-256, calculation checks, figure hashes and exported plot-data hashes. CSV files under `data/` expose the calculations. Regeneration may change SVG metadata timestamps; input/calculation reproducibility is distinct from byte-identical rendering.",
        "",
    ]
    gallery_titles = {
        "01_capture_coverage": "Capture coverage",
        "02_matched_timing": "Matched timing scenarios",
        "03_power_energy_tradeoffs": "Power and energy tradeoffs",
        "04_model_selection": "Model selection",
        "05_prospective_h200": "Prospective H200 validation",
        "06_external_device_failure": "External-device results",
        "07_feature_domains": "Feature domains",
    }
    for name, caption in CAPTIONS.items():
        lines += [
            f"## {gallery_titles[name]}",
            "",
            f"![{name}]({name}.png)",
            "",
            caption,
            "",
            f"[SVG]({name}.svg) · [PNG]({name}.png)",
            "",
        ]
    lines += [
        "## Interpretation limits",
        "",
        "Device timing comparisons differ in architecture, host CPU, compiler/toolkit and capture date. Trial ranges and correlated rows are not confidence intervals or independent sample counts. Estimated energy combines steady sensor mean power with whole-window launch rate; it is not isolated per-kernel energy. Spark's sensor is GPU-only and excludes CPU/whole-system power. Development selection, known-device prospective validation and untouched external-device challenges answer different questions. Failures and full residual tails are retained. A100 is queued and contributes no completed observation to these figures. Fitted-model feature importance is omitted because no independently archived importance vector is available and weights are deliberately not required for reproduction.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    manifest["output_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "figure-manifest.json"
    }
    (OUT / "figure-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(CAPTIONS)} PNG/SVG figure pairs and {len(INPUTS)} hashed repository inputs to {OUT}")


if __name__ == "__main__":
    main()
