import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID, BAND = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"

def use_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.prop_cycle": plt.cycler(color = PALETTE),
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.labelsize": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
    })

def lines(df, title = "", xlabel = "", ylabel = "", bands = None, marks = None,
          ax = None, figsize = (7.2, 3.6), legend_title = None):
    if ax is None:
        _, ax = plt.subplots(figsize = figsize)

    df.plot(ax = ax, legend = len(df.columns) > 1)

    blend = ax.get_xaxis_transform()
    for (lo, hi, lab) in (bands or []):
        ax.axvspan(lo, hi, color = BAND, alpha = 0.06, lw = 0)
        ax.text(lo + 0.4, 0.955, lab, transform = blend, color = MUTED, fontsize = 7,
                va = "top")
    for (xv, lab) in (marks or []):
        ax.axvline(xv, color = MUTED, lw = 0.9, alpha = 0.7)
        ax.text(xv + 0.4, 0.045, lab, transform = blend, color = MUTED, fontsize = 7,
                va = "bottom")

    ax.set(title = title, xlabel = xlabel, ylabel = ylabel)
    ax.margins(x = 0.01)
    if len(df.columns) > 1:
        ax.legend(title = legend_title, loc = "upper left", ncol = 1)
    return ax.figure

def heatmap(df, title = "", xlabel = "", ylabel = "", cbar_label = "",
            cmap = "Blues", figsize = (7.2, 5.0), annotate = False, fmt = "{:.2f}"):
    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(df.values, cmap = cmap, aspect = "auto")
    ax.set_xticks(range(len(df.columns)), df.columns)
    ax.set_yticks(range(len(df.index)), df.index)
    ax.set(title = title, xlabel = xlabel, ylabel = ylabel)
    ax.grid(False)
    if annotate:
        mid = (df.values.max() + df.values.min()) / 2
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = df.values[i, j]
                ax.text(j, i, fmt.format(v), ha = "center", va = "center", fontsize = 6.5,
                        color = "white" if v > mid else INK)
    cb = fig.colorbar(im, ax = ax, fraction = 0.03, pad = 0.02)
    cb.set_label(cbar_label, fontsize = 8, color = MUTED)
    cb.outline.set_visible(False)
    return fig

def scatter(df, x, y, hue = None, title = "", xlabel = None, ylabel = None,
            figsize = (7.2, 3.8), hline = None):
    fig, ax = plt.subplots(figsize = figsize)
    if hue:
        for i, (name, g) in enumerate(df.groupby(hue)):
            ax.scatter(g[x], g[y], s = 22, label = str(name),
                       color = PALETTE[i % len(PALETTE)],
                       edgecolor = "white", linewidth = 0.8)
        ax.legend()
    else:
        ax.scatter(df[x], df[y], s = 22, color = PALETTE[0],
                   edgecolor = "white", linewidth = 0.8)
    if hline is not None:
        ax.axhline(hline, color = MUTED, lw = 0.9, ls = "-", alpha = 0.8)
    ax.set(title = title, xlabel = xlabel or x, ylabel = ylabel or y)
    return fig
