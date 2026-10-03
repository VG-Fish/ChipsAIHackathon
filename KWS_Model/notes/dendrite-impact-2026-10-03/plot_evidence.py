"""Publication-friendly static figures from the audited CSV summaries."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def read(name: str) -> list[dict]:
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def faithful() -> None:
    wanted = {
        ("pai-faithful-b22", "none-c12x32x64"): ("3-conv base", "#667085", "o"),
        ("pai-faithful-b24", "pb-linear-max1-tanh-sw25-c12x32x64"): ("3-conv + classifier PB", "#087f8c", "*"),
        ("pai-faithful-b24", "none-c16x41x82"): ("3-conv width control (62K)", "#b1782b", "o"),
        ("pai-faithful-b24", "pb-all-max1-tanh-sw25-c12x32x64"): ("3-conv + all-layer PB", "#6057ad", "*"),
        ("pai-faithful-b24", "none-c17x50x100"): ("3-conv width control (84K)", "#b1782b", "s"),
        ("pai-faithful-b24", "none-c12x32x64x94"): ("4-conv depth control", "#d1495b", "D"),
    }
    selected = [r for r in read("faithful-arm-summary.csv") if (r["batch"], r["arm"]) in wanted]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.23, top=0.87, wspace=0.24)
    for row in selected:
        label, color, marker = wanted[(row["batch"], row["arm"])]
        y, err = float(row["test_mean_pct"]), float(row["test_sd_pct"])
        params = float(row["params_live_selected_numel_values"]) / 1000
        macs = float(row["selected_conv_linear_plus_top_macs_proxy_values"]) / 1e6
        for ax, x in zip(axes, [params, macs]):
            ax.errorbar(x, y, yerr=err, fmt=marker, markersize=10,
                        capsize=4, color=color, label=label)
            ax.grid(alpha=0.2)
            ax.set_ylabel("Held-out test accuracy (%)")
    axes[0].set_xlabel("Evaluated model parameters (thousands)")
    axes[1].set_xlabel("Affine + top-weight MAC proxy (millions)")
    axes[0].set_title("Cost in parameters")
    axes[1].set_title("Cost in affine computation")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("New faithful-loop evidence: mean ± seed SD, five seeds per model", fontsize=13)
    fig.text(0.5, 0.035, "SC2 12 classes, MFCC13+CMVN, CPU. b22 base and b24 variants; schedules differ.\nLive evaluated model counts; clean export parity unverified. MAC proxy includes top-weight products,\nomits nonlinearities, normalization, pooling, residual sums and frontend.",
             ha="center", fontsize=8)
    for ext in ["png", "svg"]:
        fig.savefig(HERE / f"faithful-accuracy-and-cost.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def lowdata() -> None:
    wanted = [
        ("sparknet-lowdata02", "pointwise_b1-bn", "sparknet_c16g16_paper"),
        ("sparknet-lowdata10", "pointwise_b1-bn", "sparknet_c16g16_paper"),
        ("sparknet-lowdata10", "pointwise_b1-bn", "sparknet_c14g16_paper"),
        ("sparknet-lowdata10", "pointwise_b2-bn", "sparknet_c16g16_paper"),
        ("sparknet-lowdata20", "pointwise_b1-bn", "sparknet_c16g16_paper"),
    ]
    lookup = {(r["family"], r["arm"], r["model"]): r for r in read("sparknet-growth-aggregates.csv")}
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = []
    for index, key in enumerate(wanted):
        row = lookup[key]
        mean = float(row["mean_gain_pp"])
        low, high = float(row["ci95_low"]), float(row["ci95_high"])
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]],
                    fmt="o", color="#087f8c", capsize=4, markersize=7)
        fraction = int(key[0][-2:])
        model = key[2].removeprefix("sparknet_").removesuffix("_paper").upper()
        arm = "block 2 + BN" if "b2-" in key[1] else "block 1 + BN"
        labels.append(f"{fraction}% data · {model} · {arm} · n={row['n']}")
    ax.axvline(0, color="#667085", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(wanted)), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Paired best-validation gain over scratch (percentage points)")
    ax.set_title("SparkNet low-data studies\nMean and descriptive paired 95% t interval", fontsize=12)
    for ext in ["png", "svg"]:
        fig.savefig(HERE / f"sparknet-lowdata-paired-effects.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    faithful()
    lowdata()
    print("Wrote two figures in PNG and editable SVG.")
