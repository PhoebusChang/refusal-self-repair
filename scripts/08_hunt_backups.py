# do dormant features hold refusal up?

import json
from contextlib import nullcontext

import torch
from scipy.stats import wilcoxon

import config
from src.model_loader import load_model
from src.prompts import load_split
from src.generation import (generate, refusal_rate, degenerate_rate,
                            refusal_score)
from src.sae import load_sae, build_bank, bank_encode
from src.hooks import ablate_features

LADDER = (2, 3, 4, 6, 8, 12, 20, 40)
N_HARM, MAXTOK = 32, 24
DEV = config.DEVICE

F0 = json.loads((config.ARTIFACTS / "F0.json").read_text())
ranked = F0["ranked"]
layers = [l for l in F0["layers"] if str(l) in ranked]

model, tokenizer = load_model()
harmful = load_split("harmful", "val")[:N_HARM]

print("building feature banks...")
banks = {}
for l in layers:
    sae = load_sae(l)
    banks[l] = build_bank(sae, ranked[str(l)]["ids"])
    del sae
    torch.cuda.empty_cache()

size = {l: banks[l]["ids"].numel() for l in layers}
rank = {l: torch.arange(size[l], device = DEV) for l in layers}

zeros = lambda: {l: torch.zeros(size[l], device = DEV) for l in layers}
count = lambda m: int(sum(v.sum().item() for v in m.values()))
merge = lambda a, b: {l: ((a[l] + b[l]) > 0).float() for l in layers}
mean = lambda v: sum(v) / len(v)

def under(mask):
    return ablate_features(model, banks, keep = mask) if mask else nullcontext()

def peaks(mask = None):
    out = zeros()

    def make(l):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            z = bank_encode(banks[l], h.to(torch.float32))
            out[l] = torch.maximum(out[l], z.flatten(0, -2).max(0).values)
        return hook

    with under(mask):
        hs = [model.model.layers[l].register_forward_hook(make(l)) for l in layers]
        try:
            generate(model, tokenizer, harmful, max_new_tokens = 4)
        finally:
            for h in hs:
                h.remove()
    return out

def measure(mask):
    with under(mask):
        s = refusal_score(model, tokenizer, harmful)
        g = generate(model, tokenizer, harmful, max_new_tokens = MAXTOK)
    return {"score": s, "asr": 1 - refusal_rate(g), "degen": degenerate_rate(g),
            "text": g}

def matched(pool, targets, z):
    cand = [(z[l][i].item(), l, i)
            for l in layers for i in pool[l].nonzero().flatten().tolist()]
    out = zeros()
    if not cand:
        return out
    vals = torch.tensor([c[0] for c in cand])
    for t in sorted(targets, reverse = True):
        j = int((vals - t).abs().argmin())
        if not vals[j].isfinite():
            break
        _, l, i = cand[j]
        out[l][i] = 1.0
        vals[j] = float("inf")            # draw without replacement
    return out

def rank_matched(pool, woke, z):
    out = zeros()
    for l in layers:
        free = pool[l].clone()
        for i in woke[l].nonzero().flatten().tolist():
            cand = free.nonzero().flatten()
            if not len(cand):
                break
            d = (cand - i).abs().float() + (z[l][cand] - z[l][i]).abs() / 1e3
            j = cand[d.argmin()]
            out[l][j], free[j] = 1.0, 0.0
    return out

def paired_p(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:                    # all deltas identically zero
        return 1.0

clean_z = peaks()
base = measure(None)
print(f"\nintact model: score {mean(base['score']):+.2f}, ASR {base['asr']:.2f}\n")

hdr = (f"{'K':>3} {'feats':>6} {'woke':>5} | {'ASR_S':>6} {'ASR_W':>6} {'ASR_C':>6} "
       f"| {'sc_S':>7} {'sc_W':>7} {'sc_C':>7} | {'comp':>6} {'p':>9} {'ctrl':>6} {'rctrl':>6} {'p_WvR':>8}")
print(hdr)
print("-" * len(hdr))

rows, detail = [], {}
for K in LADDER:
    S = {l: (rank[l] < K).float() for l in layers}
    z_S = peaks(S)
    free = {l: (S[l] == 0) for l in layers}
    woke = {l: (free[l] & (clean_z[l] == 0) & (z_S[l] > 0)).float() for l in layers}
    live = {l: (free[l] & (clean_z[l] > 0) & (z_S[l] > 0)).float() for l in layers}
    n_woke = count(woke)
    if n_woke == 0:
        print(f"{K:>3} {count(S):>6} {0:>5} |  (no dormant feature woke -- skipped)")
        continue

    tgt = [v for l in layers for v in z_S[l][woke[l] > 0].tolist()]
    ctrl = matched(live, tgt, z_S)
    ctrl_r = rank_matched(live, woke, z_S)

    m_S = measure(S)
    m_W = measure(merge(S, woke))
    m_C = measure(merge(S, ctrl))
    m_R = measure(merge(S, ctrl_r))

    comp = mean(m_S["score"]) - mean(m_W["score"])
    comp_c = mean(m_S["score"]) - mean(m_C["score"])
    comp_r = mean(m_S["score"]) - mean(m_R["score"])
    p_wr = paired_p(m_R["score"], m_W["score"])     # WOKE vs rank-matched, head to head
    p = paired_p(m_S["score"], m_W["score"])

    if K == LADDER[0]:                    # one-time no-op assertion
        mute = {l: (free[l] & (clean_z[l] == 0) & (z_S[l] == 0)).float() for l in layers}
        m_Z = measure(merge(S, matched(mute, tgt, z_S)))
        drift = mean(m_S["score"]) - mean(m_Z["score"])
        print(f"    [near-silent control: score shift {drift:+.3f} "
              f"-- {abs(drift / comp):.1%} of the woke effect]")

    print(f"{K:>3} {count(S):>6} {n_woke:>5} | {m_S['asr']:>6.2f} {m_W['asr']:>6.2f} "
          f"{m_C['asr']:>6.2f} | {mean(m_S['score']):>7.2f} {mean(m_W['score']):>7.2f} "
          f"{mean(m_C['score']):>7.2f} | {comp:>6.2f} {p:>9.2e} {comp_c:>6.2f} {comp_r:>6.2f} {p_wr:>8.1e}")

    rows.append({"K": K, "n_primaries": count(S), "n_woke": n_woke,
                 "asr_S": m_S["asr"], "asr_woke": m_W["asr"], "asr_ctrl": m_C["asr"],
                 "degen_S": m_S["degen"], "degen_woke": m_W["degen"],
                 "score_S": mean(m_S["score"]), "score_woke": mean(m_W["score"]),
                 "score_ctrl": mean(m_C["score"]),
                 "compensation": comp, "compensation_ctrl": comp_c, "p": p,
                 "compensation_rank_ctrl": comp_r, "p_woke_vs_rank_ctrl": p_wr,
                 "n_rank_ctrl": count(ctrl_r),
                 "per_prompt_S": m_S["score"], "per_prompt_woke": m_W["score"]})
    detail[K] = {"woke_completions": m_W["text"], "S_completions": m_S["text"],
                 "woke_features": sorted(
                     ({"layer": l, "feat": int(banks[l]["ids"][i]),
                       "act_after_cut": round(z_S[l][i].item(), 3)}
                      for l in layers for i in woke[l].nonzero().flatten().tolist()),
                     key = lambda d: -d["act_after_cut"])[:50]}

(config.ARTIFACTS / "hydra.json").write_text(json.dumps(
    {"ladder": LADDER, "baseline": {"score": mean(base["score"]), "asr": base["asr"]},
     "rows": rows, "detail": detail}, indent = 2))

sig = [r for r in rows if r["p"] < 0.05 and r["compensation"] > 0]
print()
if sig:
    peak = max(sig, key = lambda r: r["compensation"])
    print(f"backups carry measurable load at {len(sig)}/{len(rows)} operating points.")
    print(f"peak compensation {peak['compensation']:+.2f} log-odds at K={peak['K']} "
          f"({peak['n_woke']} woken features, p={peak['p']:.1e}),")
    print(f"vs {peak['compensation_ctrl']:+.2f} magnitude-matched, "
          f"{peak['compensation_rank_ctrl']:+.2f} rank-matched active controls.")
    wins = [r for r in rows if r["compensation"] > r["compensation_rank_ctrl"]
            and r["p_woke_vs_rank_ctrl"] < 0.05]
    print(f"WOKE beats the rank-matched control significantly at {len(wins)}/{len(rows)} K.")
else:
    print("no operating point shows significant compensation.")
if any(r["degen_woke"] > 0.15 for r in rows):
    print("WARNING: degeneracy >0.15 at some K -- that ASR is damage, not jailbreak.")
