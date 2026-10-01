# how much refusal is put back

import json

import torch
from scipy.stats import wilcoxon

import config
from src.model_loader import load_model
from src.prompts import load_split, to_chat
from src.generation import _opener_ids, REFUSAL_OPENERS, COMPLY_OPENERS
from src.sae import load_sae, build_bank

KS = (3, 8, 20, 40, 80)
DEV = config.DEVICE

F0 = json.loads((config.ARTIFACTS / "F0.json").read_text())
layers = [l for l in F0["layers"] if str(l) in F0["ranked"]]
saved = torch.load(config.ARTIFACTS / "directions.pt")
r = saved["R_hat"][saved["best_layer"], saved["best_pos_idx"]].to(DEV, torch.float32)
r = r / r.norm()

model, tokenizer = load_model()
prompts = load_split("harmful", "test")
n_layers = len(model.model.layers)

banks = {}
for l in layers:
    sae = load_sae(l)
    banks[l] = build_bank(sae, F0["ranked"][str(l)]["ids"][:max(KS)])
    del sae
    torch.cuda.empty_cache()

REF, COM = _opener_ids(tokenizer, REFUSAL_OPENERS), _opener_ids(tokenizer, COMPLY_OPENERS)
LOG = {}

def hook_for(l, K):
    b = banks.get(l)

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        x = h.to(torch.float32)
        removed = torch.zeros(x.shape[0], device = DEV)
        if K and b is not None:
            pre = x @ b["W_enc"][:, :K] + b["b_enc"][:K]
            z = pre * (pre > b["thresh"][:K])
            delta = (z @ b["W_dec"][:K]) * LOG["content"][..., None]
            removed = delta[:, -1] @ r
            x = x - delta
            h = x.to(h.dtype)
        LOG["p"][l].append((x[:, -1] @ r).cpu())
        LOG["rm"][l].append(removed.cpu())
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook

@torch.no_grad()
def run(K):
    LOG["p"] = {l: [] for l in range(n_layers)}
    LOG["rm"] = {l: [] for l in range(n_layers)}
    hs = [model.model.layers[l].register_forward_hook(hook_for(l, K)) for l in range(n_layers)]
    scores = []
    try:
        for i in range(0, len(prompts), config.BATCH_SIZE):
            enc = tokenizer([to_chat(tokenizer, p) for p in prompts[i:i + config.BATCH_SIZE]],
                            return_tensors = "pt", padding = True,
                            add_special_tokens = False).to(DEV)
            am = enc["attention_mask"].bool()
            am[torch.arange(am.shape[0]), am.float().argmax(1)] = False   # BOS
            LOG["content"] = am
            lp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
            scores += (lp[:, REF].logsumexp(-1) - lp[:, COM].logsumexp(-1)).tolist()
    finally:
        for h in hs:
            h.remove()
    p = torch.stack([torch.cat(LOG["p"][l]) for l in range(n_layers)])     # (L, N)
    rm = torch.stack([torch.cat(LOG["rm"][l]) for l in range(n_layers)])
    return p, rm, torch.tensor(scores)

p0, _, s0 = run(None)
write0 = torch.diff(p0, dim = 0, prepend = torch.zeros_like(p0[:1]))
print(f"intact: final refusal projection {p0[-1].mean():.1f}, score {s0.mean():+.2f}\n")
print(f"{'K':>3} {'removed':>8} {'shortfall':>9} {'repair':>7} | {'score':>6} "
      f"{'d_score':>8} {'p':>8} | top compensating layers (extra write onto r)")

out = {"intact_final_p": p0[-1].mean().item(), "intact_score": s0.mean().item(), "K": {}}
for K in KS:
    p, rm, s = run(K)
    removed = rm.sum(0)                           # per prompt
    shortfall = p0[-1] - p[-1]
    repair = 1 - shortfall.mean() / removed.mean()
    write = torch.diff(p, dim = 0, prepend = torch.zeros_like(p[:1])) + rm
    extra = (write - write0).mean(1)              # per layer, excluding the cut itself
    top = extra.argsort(descending = True)[:4].tolist()
    pv = wilcoxon(s.numpy(), s0.numpy()).pvalue
    print(f"{K:>3} {removed.mean():>8.1f} {shortfall.mean():>9.1f} {repair:>7.2f} | "
          f"{s.mean():>6.2f} {s.mean() - s0.mean():>+8.2f} {pv:>8.1e} | "
          + "  ".join(f"L{l}:{extra[l]:+.1f}" for l in top))
    out["K"][K] = {"removed": removed.mean().item(), "shortfall": shortfall.mean().item(),
                   "repair": repair.item(), "score": s.mean().item(),
                   "score_p_vs_intact": float(pv),
                   "extra_write_per_layer": extra.tolist(),
                   "final_p_per_prompt": p[-1].tolist()}

(config.ARTIFACTS / "repair_fraction.json").write_text(json.dumps(out, indent = 2))
