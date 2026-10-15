import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

A, OUT = config.ARTIFACTS, config.RESULTS / "figures"
OUT.mkdir(parents = True, exist_ok = True)
load = lambda n: json.loads((A / n).read_text())

BLUE, ORANGE, TEAL, GREY = "#1f6fb2", "#d9601a", "#0f8a7a", "#8a9497"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#c9cdc6", "axes.linewidth": .8,
    "axes.labelsize": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e8e3", "grid.linewidth": .7,
    "xtick.color": "#6b7478", "ytick.color": "#6b7478",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
})


def lbl(ax, x, y, t, c, dy = 3):
    ax.annotate(t, (x, y), textcoords = "offset points", xytext = (0, dy),
                ha = "center", fontsize = 7.2, color = c, fontweight = "bold")


# fig 1
rows = load("bos_judged.json")["rows"]
Ks = sorted({int(k.split("|")[2]) for k in rows if k.startswith("feat|")})
get = lambda w: [rows[f"feat|{w}|{k}"]["judged"] for k in Ks]
fig, ax = plt.subplots(figsize = (5.2, 2.5))
x = np.arange(len(Ks)); w = .26
for i, (nm, v, c) in enumerate([("all positions", get("all"), GREY),
                                ("content only", get("content"), BLUE),
                                ("BOS only", get("bos"), ORANGE)]):
    b = ax.bar(x + (i - 1) * w, v, w * .92, label = nm, color = c)
    for r_, val in zip(b, v):
        lbl(ax, r_.get_x() + r_.get_width() / 2, r_.get_height(), f"{val:.2f}", c)
ax.set_xticks(x, [f"K={k}\n({23 * k} feats)" for k in Ks])
ax.set_ylim(0, 1.18); ax.set_ylabel("judged harmful")
ax.set_title("Low-K feature ablation acts through the BOS token")
ax.grid(axis = "x", visible = False); ax.legend(loc = "upper left", ncol = 3)
fig.savefig(OUT / "fig1_bos.png"); plt.close(fig)

# fig 2
d = load("repair_fraction.json")
K = sorted(int(k) for k in d["K"])
rep = [d["K"][str(k)]["repair"] for k in K]
rem = [d["K"][str(k)]["removed"] for k in K]
mis = [d["K"][str(k)]["shortfall"] for k in K]
fig, (a1, a2) = plt.subplots(1, 2, figsize = (6.8, 2.5))
a1.plot(K, rep, "o-", color = TEAL, lw = 1.8, ms = 5)
for k, r_ in zip(K, rep):
    lbl(a1, k, r_, f"{r_:.0%}", TEAL)
a1.set(xscale = "log", ylim = (0, .88), xlabel = "features cut per layer (K)",
       ylabel = "fraction restored", title = "The model repairs most of the cut")
a1.set_xticks(K, [str(k) for k in K])
xs = np.arange(len(K))
a2.bar(xs - .19, rem, .36, label = "removed", color = ORANGE)
a2.bar(xs + .19, mis, .36, label = "still missing", color = BLUE)
a2.axhline(d["intact_final_p"], color = GREY, lw = 1.1, ls = "--")
a2.annotate(f"intact model carries {d['intact_final_p']:.0f}",
            (len(K) - .55, d["intact_final_p"]), textcoords = "offset points",
            xytext = (0, 4), ha = "right", fontsize = 7.2, color = GREY)
a2.set_xticks(xs, [str(k) for k in K])
a2.set(xlabel = "features cut per layer (K)", ylabel = "refusal signal",
       title = "More is removed than the model holds")
a2.grid(axis = "x", visible = False); a2.legend(loc = "upper left", ncol = 2)
fig.tight_layout(); fig.savefig(OUT / "fig2_repair.png"); plt.close(fig)

# fig 3
s = load("repair_specificity.json")
L = np.arange(len(s["feat K=3"]["extra_mlp"]))
fig, ax = plt.subplots(figsize = (6.8, 2.6))
ax.bar(L - .2, s["feat K=3"]["extra_mlp"], .4, label = "MLP, after refusal cut", color = TEAL)
ax.bar(L + .2, s["feat K=3"]["extra_attn"], .4, label = "attention, after refusal cut", color = BLUE)
ax.plot(L, s["rand (size of K=3)"]["extra_mlp"], color = ORANGE, lw = 1.6,
        ls = "--", label = "MLP, size-matched random shove")
ax.axhline(0, color = "#5a6366", lw = .9)
top = int(np.argmax(s["feat K=3"]["extra_mlp"]))
lbl(ax, top - .2, s["feat K=3"]["extra_mlp"][top], f"L{top}", TEAL)
ax.set(xlabel = "layer", ylabel = "extra write onto refusal direction",
       title = "Middle-layer MLPs do the repair, and it is specific")
ax.grid(axis = "x", visible = False); ax.legend(loc = "lower right")
fig.savefig(OUT / "fig3_who.png"); plt.close(fig)

# fig 4
g = load("gen_gemma.json")["conditions"]
order = [("intact", "intact"), ("cut K=3", "cut\nK=3"),
         ("cut K=3 + block MLP u (random)", "K=3 +\nrandom dir"),
         ("cut K=3 + block 6 control MLPs", "K=3 +\nwrong layers"),
         ("cut K=3 + block MLP r", "K=3 +\nblock repair"),
         ("cut K=8 + block MLP r", "K=8 +\nblock repair"),
         ("direction ablation (all layers)", "direction\nablation")]
keys = [k for k, _ in order if k in g]
names = [n for k, n in order if k in g]
asr = [g[k]["asr_keyword"] for k in keys]
jud = [g[k].get("judge", {}).get("unsafe_rate") for k in keys]
fig, ax = plt.subplots(figsize = (6.8, 2.7))
x = np.arange(len(keys))
cols = [TEAL if "block repair" in n else GREY for n in names]
b1 = ax.bar(x - .2, asr, .4, color = cols, label = "refusal skipped")
for r_, v in zip(b1, asr):
    lbl(ax, r_.get_x() + r_.get_width() / 2, v, f"{v:.2f}", "#3a4245")
if all(j is not None for j in jud):
    b2 = ax.bar(x + .2, jud, .4, color = ORANGE, label = "judged harmful")
    for r_, v in zip(b2, jud):
        lbl(ax, r_.get_x() + r_.get_width() / 2, v, f"{v:.2f}", ORANGE)
ax.set_xticks(x, names); ax.set_ylim(0, 1.18)
ax.set_ylabel("rate over 100 test prompts")
ax.set_title("Blocking the repair turns a null ablation into a jailbreak")
ax.grid(axis = "x", visible = False); ax.legend(loc = "upper left", ncol = 2)
fig.savefig(OUT / "fig4_block.png"); plt.close(fig)

# fig 5
f = load("F0.json")
lay = [l for l in f["layers"] if str(l) in f["ranked"]]
top1 = [f["ranked"][str(l)]["cos"][0] for l in lay]
fig, ax = plt.subplots(figsize = (6.8, 2.3))
ax.bar(lay, top1, .66, color = BLUE, label = "best feature at each layer")
ax.axhline(f["control_max_cos"], color = ORANGE, lw = 1.5, ls = "--",
           label = f"chance ceiling ({f['control_max_cos']:.3f})")
ax.set(xlabel = "layer", ylabel = "cos(feature, refusal direction)", ylim = (0, .42),
       title = "Refusal-aligned features exist at every depth")
ax.grid(axis = "x", visible = False); ax.legend(loc = "upper right", ncol = 2)
fig.savefig(OUT / "fig5_alignment.png"); plt.close(fig)

# fig 6
paths = sorted(A.glob("depth_*.json"))
if paths:
    fig, axes = plt.subplots(1, len(paths), figsize = (2.1 * len(paths), 2.4),
                             sharey = False)
    for ax, p in zip(np.atleast_1d(axes), paths):
        d = json.loads(p.read_text())
        rep = np.array([r["repair"] for r in d["rows"]])
        eff = np.array([r["efficiency"] for r in d["rows"]])
        ax.scatter(rep, eff, s = 13, color = BLUE, alpha = .75, edgecolors = "none")
        if len(rep) > 2:
            m, b = np.polyfit(rep, eff, 1)
            xs = np.linspace(rep.min(), rep.max(), 20)
            ax.plot(xs, m * xs + b, color = ORANGE, lw = 1.5)
        ax.set_title(f"{p.stem.replace('depth_', '')}\nr = {d['corr_repair_efficiency']:+.2f}",
                     fontsize = 8)
        ax.set_xlabel("repair fraction")
    np.atleast_1d(axes)[0].set_ylabel("behaviour per unit damage")
    fig.suptitle("The more a model repairs, the more blocking the repair buys",
                 x = .01, ha = "left", fontsize = 9.5, fontweight = "bold")
    fig.tight_layout(); fig.savefig(OUT / "fig6_dose_response.png"); plt.close(fig)

print("figures ->", OUT)
for p in sorted(OUT.glob("fig*.png")):
    print(" ", p.name, f"{p.stat().st_size // 1024} KB")
