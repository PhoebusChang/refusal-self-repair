# is the wake-up targeted?

import json
import sys

import torch
from scipy.stats import mannwhitneyu

import config
from src.model_loader import load_model
from src.prompts import load_split, to_chat
from src.generation import _opener_ids, REFUSAL_OPENERS, COMPLY_OPENERS
from src.sae import load_sae, build_bank, align

KS = (2, 8)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
CHUNK = 6                                   # full SAEs resident at once
DEV = config.DEVICE

saved = torch.load(config.ARTIFACTS / "directions.pt")
r = saved["R_hat"][saved["best_layer"], saved["best_pos_idx"]].to(DEV, torch.float32)
r = r / r.norm()
F0 = json.loads((config.ARTIFACTS / "F0.json").read_text())
layers = [l for l in F0["layers"] if str(l) in F0["ranked"]]

model, tokenizer = load_model()
prompts = load_split("harmful", SPLIT)

print("banks + full-SAE alignment...")
banks, cos = {}, {}
g = torch.Generator(device = "cpu").manual_seed(config.SEED)
u = {}
for l in layers:
    sae = load_sae(l)
    banks[l] = build_bank(sae, F0["ranked"][str(l)]["ids"][:max(KS)])
    cos[l] = align(sae, r).cpu()
    v = torch.randn(r.numel(), generator = g).to(DEV)
    v = v - (v @ r) * r                     # random, orthogonal to refusal
    u[l] = v / v.norm()
    del sae
    torch.cuda.empty_cache()

def edit_hooks(mode, K):
    def make(l):
        b = banks[l]
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            x = h.to(torch.float32)
            pre = x[..., None, :] @ b["W_enc"][:, :K]
            z = (pre + b["b_enc"][:K]).squeeze(-2)
            z = z * (z > b["thresh"][:K])
            delta = z @ b["W_dec"][:K]
            delta = delta * VALID["m"][..., None]       # never touch BOS / pad
            if mode == "cut":
                step = -delta
            elif mode == "flip":
                step = delta
            else:
                step = -delta.norm(dim = -1, keepdim = True) * u[l]
            m = VALID["m"]
            SIZE.append((step.norm(dim = -1)[m] / x.norm(dim = -1)[m]).mean().item())
            x = x + step
            h = x.to(h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return hook
    return [model.model.layers[l].register_forward_hook(make(l)) for l in layers]

VALID, SIZE = {}, []

def batches():
    for i in range(0, len(prompts), config.BATCH_SIZE):
        texts = [to_chat(tokenizer, p) for p in prompts[i:i + config.BATCH_SIZE]]
        enc = tokenizer(texts, return_tensors = "pt", padding = True,
                        add_special_tokens = False).to(DEV)
        am = enc["attention_mask"].bool()
        am[torch.arange(am.shape[0]), am.float().argmax(1)] = False   # drop BOS
        VALID["m"] = am
        yield enc

@torch.no_grad()
def margins(cond, chunk_saes):
    out = {l: torch.full((sae.W_enc.shape[1],), -1e9, device = DEV)
           for l, sae in chunk_saes.items()}

    def rec(l, sae):
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output).to(torch.float32)
            m = (h @ sae.W_enc + sae.b_enc - sae.threshold)
            m = m.masked_fill(~VALID["m"][..., None], -1e9)
            out[l] = torch.maximum(out[l], m.flatten(0, 1).max(0).values)
        return hook

    hs = edit_hooks(*cond) if cond else []
    hs += [model.model.layers[l].register_forward_hook(rec(l, s))
           for l, s in chunk_saes.items()]
    try:
        for enc in batches():
            model(**enc)
    finally:
        for h in hs:
            h.remove()
    return {l: v.cpu() for l, v in out.items()}

REF, COM = _opener_ids(tokenizer, REFUSAL_OPENERS), _opener_ids(tokenizer, COMPLY_OPENERS)

@torch.no_grad()
def score(cond):
    hs = edit_hooks(*cond) if cond else []
    SIZE.clear()
    out = []
    try:
        for enc in batches():
            lp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
            out += (lp[:, REF].logsumexp(-1) - lp[:, COM].logsumexp(-1)).tolist()
    finally:
        for h in hs:
            h.remove()
    return out, (sum(SIZE) / len(SIZE) if SIZE else 0.0)

conds = [None] + [(m, K) for K in KS for m in ("cut", "rand", "flip")]
name = lambda c: "clean" if c is None else f"{c[0]}@K{c[1]}"
M = {name(c): {} for c in conds}
for i in range(0, len(layers), CHUNK):
    chunk = {l: load_sae(l) for l in layers[i:i + CHUNK]}
    print(f"  recording layers {list(chunk)}")
    for c in conds:
        M[name(c)].update(margins(c, chunk))
    del chunk
    torch.cuda.empty_cache()

mean = lambda v: sum(v) / len(v)
S_scores, S_size = {}, {}
for c in conds:
    S_scores[name(c)], S_size[name(c)] = score(c)
print(f"\n{SPLIT} split, {len(prompts)} harmful prompts\n")

cat = lambda d: torch.cat([d[l] for l in layers])
cos_all = cat(cos)
top1 = cat({l: cos[l] >= cos[l].quantile(0.99) for l in layers})
clean = cat(M["clean"])
silent = clean <= 0
base_aligned = top1[silent].float().mean().item()

report = {"split": SPLIT, "n_prompts": len(prompts), "base_frac_aligned": base_aligned,
          "score": {k: mean(v) for k, v in S_scores.items()}, "K": {}}
print(f"baseline: of {int(silent.sum())} features silent on the intact model, "
      f"{base_aligned:.2%} are refusal-aligned (top 1% cosine in their layer)\n")
print(f"{'cond':>9} {'score':>7} {'woke':>6} {'aligned%':>9} {'enrich':>7} "
      f"{'meancos':>8} {'clean margin':>13} {'|step|/|h|':>11}")
print(f"{'clean':>9} {mean(S_scores['clean']):>7.2f}")
for K in KS:
    woke = {}
    for m in ("cut", "rand", "flip"):
        w = silent & (cat(M[f"{m}@K{K}"]) > 0)
        woke[m] = w
        fa = top1[w].float().mean().item() if w.any() else 0.0
        print(f"{m + '@K' + str(K):>9} {mean(S_scores[f'{m}@K{K}']):>7.2f} "
              f"{int(w.sum()):>6} {fa:>9.2%} {fa / base_aligned:>7.1f}x "
              f"{cos_all[w].mean().item() if w.any() else 0:>8.3f} "
              f"{clean[w].median().item() if w.any() else 0:>13.2f} "
              f"{S_size[f'{m}@K{K}']:>11.3f}")
    c, rnd = woke["cut"], woke["rand"]
    jac = (c & rnd).sum().item() / max((c | rnd).sum().item(), 1)
    p = mannwhitneyu(cos_all[c], cos_all[rnd], alternative = "greater").pvalue\
        if c.any() and rnd.any() else 1.0
    print(f"          Jaccard(cut, rand) = {jac:.2f}   "
          f"cos(cut-woke) > cos(rand-woke): p = {p:.1e}")
    print(f"          silent-feature clean margin median: {clean[silent].median():.2f}\n")
    report["K"][K] = {m: {"n": int(woke[m].sum()),
                          "frac_aligned": top1[woke[m]].float().mean().item()
                          if woke[m].any() else 0.0,
                          "mean_cos": cos_all[woke[m]].mean().item() if woke[m].any() else 0.0,
                          "median_clean_margin": clean[woke[m]].median().item()
                          if woke[m].any() else 0.0}
                      for m in woke}
    report["K"][K]["jaccard_cut_rand"] = jac
    report["K"][K]["p_cut_more_aligned_than_rand"] = float(p)

(config.ARTIFACTS / f"wake_mechanism_{SPLIT}.json").write_text(json.dumps(report, indent = 2))
