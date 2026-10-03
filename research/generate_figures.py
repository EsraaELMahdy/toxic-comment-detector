from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INK = "#0f172a"
MUTED = "#64748b"
INDIGO = "#6366f1"
CYAN = "#0ea5e9"
AMBER = "#f59e0b"
ROSE = "#e11d48"
GREEN = "#10b981"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _finish(axis, title: str) -> None:
    axis.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=12)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def figure_memory_footprint() -> Path:
    parameters = 7e9
    precisions = [
        ("FP32\n(32-bit)", 32, INDIGO),
        ("FP16 / BF16\n(16-bit)", 16, CYAN),
        ("INT8\n(8-bit)", 8, GREEN),
        ("INT4 / NF4\n(4-bit)", 4, AMBER),
    ]

    labels = [name for name, _, _ in precisions]
    memory_gb = [parameters * bits / 8 / 1e9 for _, bits, _ in precisions]
    colours = [colour for _, _, colour in precisions]

    figure, axis = plt.subplots(figsize=(8.4, 4.5))
    bars = axis.bar(labels, memory_gb, color=colours, width=0.58, zorder=3)

    for bar, value in zip(bars, memory_gb):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.2f} GB",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )

    baseline = memory_gb[0]
    for bar, value in zip(bars[1:], memory_gb[1:]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            f"÷{baseline / value:.0f}\n{value / baseline:.0%} of FP32",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    axis.set_ylabel("Weight memory (GB)", fontsize=10.5)
    axis.set_ylim(0, max(memory_gb) * 1.18)
    _finish(axis, "Memory footprint of a 7B-parameter model   M = N × b / 8")

    figure.tight_layout()
    path = FIGURES_DIR / "01_memory_footprint.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def quantize_dequantize(values: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    levels = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(values)) / levels
    integers = np.clip(np.round(values / scale), -levels - 1, levels)
    return integers * scale, scale


def figure_quantization_error() -> Path:
    rng = np.random.default_rng(42)
    x = np.arange(96)
    original = 2.4 * np.sin(x / 7.6) * np.exp(-x / 55) + 0.35 * rng.standard_normal(x.size)

    reconstructed, scale = quantize_dequantize(original, bits=8)
    residual = original - reconstructed

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 1.15]}
    )

    top.plot(x, original, "-", color=INDIGO, linewidth=2.2, label="Original (FP32)", zorder=3)
    top.step(
        x,
        reconstructed,
        where="mid",
        color=ROSE,
        linewidth=1.4,
        alpha=0.9,
        label=f"Reconstructed (INT8, step = {scale:.3f})",
        zorder=4,
    )
    top.set_ylabel("Value", fontsize=10.5)
    top.legend(frameon=False, fontsize=9.5, loc="upper right")
    _finish(top, "INT8 quantization: the staircase is the precision you give up")

    bottom.bar(x, residual, width=1.0, color=CYAN, alpha=0.85, zorder=3)
    bottom.axhline(0, color=MUTED, linewidth=0.9)
    bottom.set_xlabel("Sample index", fontsize=10.5)
    bottom.set_ylabel("Error", fontsize=10.5)
    _finish(bottom, f"Residual  ·  max |error| = {np.max(np.abs(residual)):.4f}")

    figure.tight_layout()
    path = FIGURES_DIR / "02_quantization_error.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_compression_ratio() -> Path:
    scenarios = [
        ("FP32", 1, 100.0, INDIGO),
        ("FP16", 2, 99.9, CYAN),
        ("INT8", 4, 99.4, GREEN),
        ("INT4", 8, 97.5, AMBER),
    ]

    labels = [name for name, _, _, _ in scenarios]
    ratios = [ratio for _, ratio, _, _ in scenarios]
    quality = [score for _, _, score, _ in scenarios]
    colours = [colour for _, _, _, colour in scenarios]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    bars = left.bar(labels, ratios, color=colours, width=0.6, zorder=3)
    for bar, ratio in zip(bars, ratios):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + 0.22,
            f"{ratio}×",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    left.set_ylabel("Compression vs FP32", fontsize=10.5)
    left.set_ylim(0, max(ratios) * 1.22)
    _finish(left, "Compression ratio   ratio = b_fp32 / b_target")

    bars = right.bar(labels, quality, color=colours, width=0.6, zorder=3)
    for bar, score in zip(bars, quality):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.18,
            f"{score:.1f}%",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
    right.set_ylabel("Retained accuracy (illustrative)", fontsize=10.5)
    right.set_ylim(94, 101.4)
    _finish(right, "Quality cost: the cheaper the storage, the sharper the drop")

    figure.tight_layout()
    path = FIGURES_DIR / "03_compression_ratio.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    for builder in (figure_memory_footprint, figure_quantization_error, figure_compression_ratio):
        print(f"wrote {builder()}")


if __name__ == "__main__":
    main()
