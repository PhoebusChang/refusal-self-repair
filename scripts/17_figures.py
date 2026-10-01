# figures

import json

import matplotlib.pyplot as plt
import numpy as np

import config
from src.plots import use_style, PALETTE, MUTED, INK

use_style()
A, OUT = config.ARTIFACTS, config.RESULTS / "figures"
OUT.mkdir(parents = True, exist_ok = True)
load = lambda name: json.loads((A / name).read_text())

def bars(ax, labels, series, colors, ylabel, ylim = None):
    w = 0.8 / len(series)
    x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label = name,
                   color = colors[i], edgecolor = "white", linewidth = 0.6)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                    f"{v:.2f}", ha = "center", va = "bottom", fontsize = 6.5, color = INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis = "x", visible = False)
    ax.legend(loc = "upper left")

# 1. BOS decomposition
rows = load("bos_check.json")
feat = [r for r in rows if r["kind"] == "feat"]
Ks = sorted({r["K"] for r in feat})
get = lambda w: [next(r["asr"] for r in feat if r["K"] == k and r["where"] == w) for k in Ks]
fig, ax = plt.subplots(figsize = (7.2, 3.4))
bars(ax, [f"K={k}\n({23 * k} feats)" for k in Ks],
     {"all positions": get("all"), "content only": get("content"), "BOS only": get("bos")},
     PALETTE[:3], "attack success rate (keyword)", (0, 1.12))
ax.set_title("Low-K feature ablation only jailbreaks when BOS is edited too")
fig.savefig(OUT / "fig1_bos.png")

# 2. repair curve
d = load("repair_fraction.json")
K = sorted(int(k) for k in d["K"])
rep = [d["K"][str(k)]["repair"] for k in K]
ds = [d["K"][str(k)]["score"] - d["intact_score"] for k in K]
fig, (a1, a2) = plt.subplots(1, 2, figsize = (7.2, 3.0))
a1.plot(K, rep, marker = "o", color = PALETTE[0])
a1.set(xscale = "log", xlabel = "features cut per layer (K)", ylabel = "fraction repaired",
       ylim = (0, 1), title = "The model repairs most of the cut")
a1.set_xticks(K, [str(k) for k in K])
a2.plot(K, ds, marker = "o", color = PALETTE[1])
a2.axhline(0, color = MUTED, lw = 0.9)
a2.set(xscale = "log", xlabel = "features cut per layer (K)",
       ylabel = "refusal score change vs intact", title = "…and overshoots at small K")
a2.set_xticks(K, [str(k) for k in K])
fig.tight_layout()
fig.savefig(OUT / "fig2_repair_curve.png")

# 3. who repairs
d = load("repair_specificity.json")
fig, ax = plt.subplots(figsize = (7.2, 3.2))
layers = np.arange(len(d["feat K=3"]["extra_mlp"]))
ax.plot(layers, d["feat K=3"]["extra_mlp"], marker = "o", color = PALETTE[0],
        label = "MLP, after refusal cut")
ax.plot(layers, d["feat K=3"]["extra_attn"], marker = "o", color = PALETTE[1],
        label = "attention, after refusal cut")
ax.plot(layers, d["rand (size of K=3)"]["extra_mlp"], ls = "--", color = PALETTE[0],
        alpha = 0.55, label = "MLP, after same-size random shove")
ax.axhline(0, color = MUTED, lw = 0.9)
ax.set(xlabel = "layer", ylabel = "extra write onto refusal direction",
       title = "The repair is done by middle-layer MLPs, and it is targeted")
ax.legend(loc = "upper left")
fig.savefig(OUT / "fig3_who_repairs.png")

d = load("gen_gemma.json")["conditions"]
order = ["intact", "cut K=3", "cut K=3 + block MLP u (random)", "cut K=3 + block MLP r",
         "cut K=8", "cut K=8 + block MLP u (random)", "cut K=8 + block MLP r",
         "direction ablation (all layers)"]
short = ["intact", "cut\nK=3", "K=3 +\nblock u", "K=3 +\nblock r", "cut\nK=8",
         "K=8 +\nblock u", "K=8 +\nblock r", "direction\nablation"]
series = {"refusal skipped (keyword)": [d[k]["asr_keyword"] for k in order]}
if all("judge" in d[k] for k in order):
    series["harmful per Llama Guard"] = [d[k]["judge"]["unsafe_rate"] for k in order]
fig, ax = plt.subplots(figsize = (7.2, 3.4))
bars(ax, short, series, PALETTE[:2], "rate over 100 test prompts", (0, 1.15))
ax.set_title("Blocking the MLP repair along r turns a null cut into a jailbreak")
fig.savefig(OUT / "fig4_block.png")

print("figures ->", OUT)
