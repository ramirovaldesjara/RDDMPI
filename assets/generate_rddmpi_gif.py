from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

out = Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=True)
gif_path = out / "rddmpi_demo.gif"

x = np.linspace(0, 10, 180)
truth = 0.55 * np.sin(1.2 * x) + 0.22 * np.sin(2.5 * x + 0.5) + 0.04 * x
gap = (x >= 4.0) & (x <= 6.2)

observed = truth.copy()
observed[gap] = np.nan

baseline = 0.50 * np.sin(1.2 * x) + 0.10 * np.sin(2.5 * x + 0.5) + 0.04 * x
baseline_full = truth.copy()
baseline_full[gap] = baseline[gap]

rng = np.random.default_rng(8)
true_residual = truth - baseline

perturbations = []
for _ in range(12):
    phase = rng.uniform(-0.7, 0.7)
    amp = rng.uniform(0.025, 0.06)
    perturbation = (
        amp * np.sin(3.0 * x + phase)
        + rng.normal(0, 0.01, len(x))
    )
    perturbations.append(perturbation)

perturbations = np.stack(perturbations)
perturbations -= perturbations.mean(axis=0, keepdims=True)

samples = []
for perturbation in perturbations:
    residual = true_residual + perturbation
    sample = baseline_full.copy()
    sample[gap] = baseline[gap] + residual[gap]
    samples.append(sample)

frames = []


def add_frame(stage, n_samples=0, show_band=False, hold=1):
    fig, ax = plt.subplots(figsize=(9, 4.3), dpi=100)

    # NaNs break the observed line cleanly across the missing interval.
    ax.plot(x, observed, linewidth=2.2, label="Observed")
    ax.plot(
        x[gap],
        truth[gap],
        linestyle="--",
        alpha=0.35,
        linewidth=1.6,
        label="Ground Truth",
    )

    if stage >= 1:
        ax.plot(
            x[gap],
            baseline[gap],
            linewidth=2.4,
            label="Deterministic baseline",
        )

    if stage >= 2 and n_samples:
        for i in range(n_samples):
            ax.plot(x[gap], samples[i][gap], alpha=0.28, linewidth=1.15)

    if show_band:
        arr = np.stack([sample[gap] for sample in samples])
        lo = np.quantile(arr, 0.10, axis=0)
        hi = np.quantile(arr, 0.90, axis=0)
        mean = arr.mean(axis=0)
        ax.fill_between(
            x[gap], lo, hi, alpha=0.18, label="Probabilistic interval"
        )
        ax.plot(x[gap], mean, linewidth=2.2, label="Sample mean")

    ax.axvspan(4.0, 6.2, alpha=0.08)
    ax.set_xlim(0, 10)
    ax.set_ylim(truth.min() - 0.25, truth.max() + 0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(
        "RDDMPI — Residual Denoising Diffusion Model for Probabilistic "
        "Multivariate Time Series Imputation",
        pad=12,
    )

    if stage == 0:
        subtitle = "1. Incomplete multivariate time series"
    elif stage == 1:
        subtitle = "2. A deterministic model provides an initial imputation"
    elif stage == 2 and not show_band:
        subtitle = f"3. Diffusion models residual uncertainty ({n_samples} samples)"
    else:
        subtitle = "4. Baseline + sampled residuals → probabilistic imputations"

    ax.text(
        0.5,
        0.94,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.12),
    )

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    ax.legend(
        seen.values(),
        seen.keys(),
        loc="lower left",
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE)
    for _ in range(hold):
        frames.append(img.copy())


add_frame(0, hold=5)
add_frame(1, hold=5)
for n in [1, 2, 4, 6, 9, 12]:
    add_frame(2, n_samples=n, hold=2)
add_frame(3, n_samples=12, show_band=True, hold=7)

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=260,
    loop=0,
    optimize=True,
)
print(gif_path)
