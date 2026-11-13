"""Publication figure helpers following figures4papers scientific-figure-making.

The conventions (palette, spines, export) are the ones specified in
Chen Liu's figures4papers skill: scientific-figure-making/references/api.md
and design-theory.md. Helpers are implemented here so the report figures
do not depend on that repository at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gray_dark": "#4D4D4D",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size,
            "xtick.labelsize": max(style.font_size - 2, 8),
            "ytick.labelsize": max(style.font_size - 2, 8),
            "legend.fontsize": max(style.font_size - 2, 8),
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": style.axes_linewidth,
            "xtick.major.width": style.axes_linewidth * 0.6,
            "ytick.major.width": style.axes_linewidth * 0.6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": bool(style.use_tex),
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.axisbelow": True,
        }
    )


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def finalize_figure(fig, out_path, formats=None, dpi: int = 300, close: bool = True, pad: float = 0.05, **kwargs):
    path = Path(out_path)
    if formats is None:
        formats = [path.suffix.lstrip(".")] if path.suffix else ["pdf", "png"]
    stem = path.with_suffix("") if path.suffix else path
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    saved = []
    for fmt in formats:
        dest = stem.with_suffix("." + fmt)
        fig.savefig(dest, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        saved.append(dest)
    if close:
        plt.close(fig)
    return saved


def _as_1d(name, values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None, annotate=False, hatch=None):
    categories = list(categories)
    series = [np.asarray(s, dtype=float) for s in series]
    if any(len(s) != len(categories) for s in series):
        raise ValueError("each series must have one value per category")
    colors = list(colors) if colors is not None else DEFAULT_COLORS
    n_series = len(series)
    x = np.arange(len(categories))
    width = min(0.8 / max(n_series, 1), 0.36)
    last = None
    for i, (vals, label) in enumerate(zip(series, labels)):
        offset = (i - (n_series - 1) / 2) * width
        last = ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            label=label,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=1.2,
            hatch=None if hatch is None else hatch[i % len(hatch)],
            zorder=3,
        )
        if annotate:
            annotate_bars(ax, last, fmt="{:.2f}", fontsize=max(ax.yaxis.label.get_size() - 6, 8))
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.legend()
    return last


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=10, padding=3):
    for rect in bars:
        height = rect.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None, show_shadow=True, yerr=None, markers=None):
    x = _as_1d("x", x)
    colors = list(colors) if colors is not None else DEFAULT_COLORS
    markers = markers or ["o", "s", "D", "^", "v", "P"]
    for i, (y, label) in enumerate(zip(y_series, labels)):
        y = _as_1d(label, y)
        if len(y) != len(x):
            raise ValueError(f"{label} length {len(y)} != x length {len(x)}")
        color = colors[i % len(colors)]
        ax.plot(x, y, color=color, lw=2.4, marker=markers[i % len(markers)], ms=7, label=label, zorder=3)
        if show_shadow and yerr is not None:
            err = np.asarray(yerr[i], dtype=float)
            if err.ndim == 1:
                low, high = y - err, y + err
            else:
                low, high = y - err[0], y + err[1]
            ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0, zorder=2)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap="magma", cbar_label=None, annotate=False, vmin=None, vmax=None):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("heatmap matrix must be 2D")
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    if annotate:
        finite = matrix[np.isfinite(matrix)]
        threshold = np.nanmedian(finite) if finite.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isfinite(val):
                    continue
                color = "white" if val >= threshold else "black"
                ax.text(j, i, f"{val:.0f}" if abs(val) >= 10 else f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return image


def make_scatter(ax, x, y, label=None, color=None, size=50, alpha=0.7, edgecolors="none"):
    x = _as_1d("x", x)
    y = _as_1d("y", y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    color = PALETTE["blue_main"] if color is None else color
    ax.scatter(x, y, s=size, c=color, alpha=alpha, label=label, edgecolors=edgecolors, linewidths=0.3, zorder=3)
    if label:
        ax.legend()
    return ax
