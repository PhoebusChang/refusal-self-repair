import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

import torch
from scipy.stats import wilcoxon

import config
from src.model_loader import load_model
from src.prompts import load_split, to_chat
from src.generation import _opener_ids, REFUSAL_OPENERS, COMPLY_OPENERS
from src.sae import load_sae, build_bank

K = 3
DIR_LAYERS = (5, 9, 13)
DEV = config.DEVICE

F0 = json.loads((config.ARTIFACTS / "F0.json").read_text())
sae_layers = [l for l in F0["layers"] if str(l) in F0["ranked"]]
saved = torch.load(config.ARTIFACTS / "directions.pt")
r = saved["R_hat"][saved["best_layer"], saved["best_pos_idx"]].to(DEV, torch.float32)
r = r / r.norm()

model, tokenizer = load_model()
prompts = load_split("harmful", "test")
L = model.model.layers
n_layers = len(L)

banks = {}
for l in sae_layers:
    sae = load_sae(l)
    banks[l] = build_bank(sae, F0["ranked"][str(l)]["ids"][:K])
    del sae
    torch.cuda.empty_cache()

g = torch.Generator().manual_seed(config.SEED)
u = {}
for l in sae_layers:
    v = torch.randn(r.numel(), generator = g).to(DEV)
    v = v - (v @ r) * r
    u[l] = v / v.norm()

REF, COM = _opener_ids(tokenizer, REFUSAL_OPENERS), _opener_ids(tokenizer, COMPLY_OPENERS)
S = {}

def edit(l, mode):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        x = h.to(torch.float32)
        m = S["content"][..., None]
        if mode in ("feat", "rand") and l in banks:
            b = banks[l]
            pre = x @ b["W_enc"] + b["b_enc"]
            delta = ((pre * (pre > b["thresh"])) @ b["W_dec"]) * m
            if mode == "rand":
                delta = delta.norm(dim = -1, keepdim = True) * u[l]
        elif mode == ("dir", l):
            delta = (x @ r)[..., None] * r * m
        else:
            return output
        S["rm"][l].append((delta[:, -1] @ r).cpu())
        h = (x - delta).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook

def tap(l, part):
    def hook(module, args, output):
        S[part][l].append((output[:, -1].float() @ r).cpu())
    return hook

@torch.no_grad()
def run(mode):
    for k in ("attn", "mlp", "rm", "p"):
        S[k] = {l: [] for l in range(n_layers)}
    hs = []
    for l in range(n_layers):
        hs.append(L[l].post_attention_layernorm.register_forward_hook(tap(l, "attn")))
        hs.append(L[l].post_feedforward_layernorm.register_forward_hook(tap(l, "mlp")))
        if mode is not None:
            hs.append(L[l].register_forward_hook(edit(l, mode)))
    hs.append(L[-1].register_forward_hook(
        lambda mod, a, o: S["p"][0].append(((o[0] if isinstance(o, tuple) else o)[:, -1].float() @ r).cpu())))
    scores = []
    try:
        for i in range(0, len(prompts), config.BATCH_SIZE):
            enc = tokenizer([to_chat(tokenizer, p) for p in prompts[i:i + config.BATCH_SIZE]],
                            return_tensors = "pt", padding = True,
                            add_special_tokens = False).to(DEV)
            am = enc["attention_mask"].bool()
            am[torch.arange(am.shape[0]), am.float().argmax(1)] = False
            S["content"] = am
            lp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
            scores += (lp[:, REF].logsumexp(-1) - lp[:, COM].logsumexp(-1)).tolist()
    finally:
        for h in hs:
            h.remove()
    stack = lambda d: torch.stack([torch.cat(d[l]) if d[l] else torch.zeros(len(prompts))
                                   for l in range(n_layers)])
    return {"attn": stack(S["attn"]), "mlp": stack(S["mlp"]), "rm": stack(S["rm"]),
            "p": torch.cat(S["p"][0]), "score": torch.tensor(scores)}

base = run(None)
print(f"intact: final p {base['p'].mean():.1f}, score {base['score'].mean():+.2f}\n")
modes = [("feat", "feat K=3"), ("rand", "rand (size of K=3)")] +\
        [(("dir", l), f"dir @ L{l}") for l in DIR_LAYERS]

print(f"{'condition':>20} {'removed':>8} {'missing':>8} {'repair':>7} | "
      f"{'xtra attn':>9} {'xtra mlp':>9} | {'d_score':>8} {'p':>8} | top layers (attn+mlp)")
out = {}
for mode, label in modes:
    o = run(mode)
    removed = o["rm"].sum(0).mean().item()
    missing = (base["p"] - o["p"]).mean().item()
    xa = (o["attn"] - base["attn"]).mean(1)
    xm = (o["mlp"] - base["mlp"]).mean(1)
    tot = xa + xm
    top = tot.argsort(descending = True)[:4].tolist()
    ds = (o["score"] - base["score"]).mean().item()
    pv = wilcoxon(o["score"].numpy(), base["score"].numpy()).pvalue
    rep = 1 - missing / removed if abs(removed) > 1e-6 else float("nan")
    print(f"{label:>20} {removed:>8.1f} {missing:>8.1f} {rep:>7.2f} | "
          f"{xa.sum():>+9.1f} {xm.sum():>+9.1f} | {ds:>+8.2f} {pv:>8.1e} | "
          + "  ".join(f"L{l}:{tot[l]:+.1f}" for l in top))
    out[label] = {"removed": removed, "missing": missing, "repair": rep,
                  "extra_attn_total": xa.sum().item(), "extra_mlp_total": xm.sum().item(),
                  "extra_attn": xa.tolist(), "extra_mlp": xm.tolist(),
                  "d_score": ds, "p": float(pv)}

(config.ARTIFACTS / "repair_specificity.json").write_text(json.dumps(out, indent = 2))
