# McNemar + bootstrap CIs

import json

import numpy as np
from scipy.stats import binomtest

import config

rng = np.random.default_rng(config.SEED)

def mcnemar(a, b):
    a, b = np.array(a), np.array(b)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if x + y == 0 else binomtest(x, x + y, 0.5).pvalue

def ci(flags):
    f = np.array(flags, dtype = float)
    boots = rng.choice(f, (5000, len(f))).mean(1)
    return np.percentile(boots, [2.5, 97.5])

def find(conds, *must, avoid = ()):
    return next(k for k in conds if all(m in k for m in must) and not any(v in k for v in avoid))

lines = ["| model / cut | cut | + block MLP r | + block MLP u | p (r vs cut) | p (r vs u) |",
         "|---|---|---|---|---|---|"]
sets = [("gen_gemma.json", [("K=3", "cut K=3"), ("K=8", "cut K=8")])]
out = {}
for name, cuts in sets:
    d = json.loads((config.ARTIFACTS / name).read_text())
    c = d["conditions"]
    for spec in cuts:
        if spec:
            tag, base = spec
            k_cut = base
            k_r, k_u = f"{base} + block MLP r", f"{base} + block MLP u (random)"
        else:
            tag = d["gen_design"] + " (no SAE)"
            k_cut = find(c, "cut ", avoid = ("block",))
            k_r, k_u = find(c, "block MLP r"), find(c, "block MLP u")
        F = {k: c[k]["judge"]["flags"] for k in (k_cut, k_r, k_u)}
        rate = {k: np.mean(v) for k, v in F.items()}
        p1, p2 = mcnemar(F[k_r], F[k_cut]), mcnemar(F[k_r], F[k_u])
        fmt = lambda k: f"{rate[k]:.2f} [{ci(F[k])[0]:.2f}, {ci(F[k])[1]:.2f}]"
        model = d.get("model", "").split("/")[-1]
        lines.append(f"| {model} {tag} | {fmt(k_cut)} | **{fmt(k_r)}** | {fmt(k_u)} | "
                     f"{p1:.1e} | {p2:.1e} |")
        out[f"{model} {tag}"] = {"cut": rate[k_cut], "block_r": rate[k_r],
                                 "block_u": rate[k_u], "p_r_vs_cut": p1, "p_r_vs_u": p2}

table = "\n".join(lines)
print(table)
(config.RESULTS / "stats_table.md").write_text(table + "\n")
(config.ARTIFACTS / "stats.json").write_text(json.dumps(out, indent = 2))
