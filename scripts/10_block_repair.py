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

REPAIR = (9, 11, 12, 14, 16, 17)       # top compensators from 08/09
CONTROL = (19, 20, 21, 22, 23, 24)
DEV = config.DEVICE

F0 = json.loads((config.ARTIFACTS / "F0.json").read_text())
sae_layers = [l for l in F0["layers"] if str(l) in F0["ranked"]]
saved = torch.load(config.ARTIFACTS / "directions.pt")
r = saved["R_hat"][saved["best_layer"], saved["best_pos_idx"]].to(DEV, torch.float32)
r = r / r.norm()
model, tokenizer = load_model()
prompts = load_split("harmful", "test")
L = model.model.layers

banks = {}
for l in sae_layers:
    sae = load_sae(l)
    banks[l] = build_bank(sae, F0["ranked"][str(l)]["ids"][:20])
    del sae
    torch.cuda.empty_cache()

REF, COM = _opener_ids(tokenizer, REFUSAL_OPENERS), _opener_ids(tokenizer, COMPLY_OPENERS)
REF_SET = set(REF)
S = {}

def cut(l, K):
    b = banks[l]
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        x = h.to(torch.float32)
        pre = x @ b["W_enc"][:, :K] + b["b_enc"][:K]
        z = pre * (pre > b["thresh"][:K])
        h = (x - (z @ b["W_dec"][:K]) * S["content"][..., None]).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook

def record(key):
    def hook(module, args, output):
        S["cache"][key] = output.detach()
    return hook

def freeze(key, only_r):
    def hook(module, args, output):
        clean = S["cache"][key]
        if not only_r:
            return clean
        diff = ((output.float() - clean.float()) @ r) * S["content"]
        return (output.float() - diff[..., None] * r).to(output.dtype)
    return hook

def part(l, which):
    return L[l].post_feedforward_layernorm if which == "mlp" else L[l].post_attention_layernorm

@torch.no_grad()
def run(K = None, frozen = (), which = "mlp", only_r = False):
    scores, first_ref = [], []
    for i in range(0, len(prompts), config.BATCH_SIZE):
        enc = tokenizer([to_chat(tokenizer, p) for p in prompts[i:i + config.BATCH_SIZE]],
                        return_tensors = "pt", padding = True,
                        add_special_tokens = False).to(DEV)
        am = enc["attention_mask"].bool()
        am[torch.arange(am.shape[0]), am.float().argmax(1)] = False
        S["content"], S["cache"] = am, {}
        hs = [part(l, which).register_forward_hook(record(l)) for l in frozen]
        try:
            model(**enc)                                     # intact reference pass
        finally:
            for h in hs:
                h.remove()
        hs = [part(l, which).register_forward_hook(freeze(l, only_r)) for l in frozen]
        if K:
            hs += [L[l].register_forward_hook(cut(l, K)) for l in sae_layers]
        try:
            lp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
        finally:
            for h in hs:
                h.remove()
        scores += (lp[:, REF].logsumexp(-1) - lp[:, COM].logsumexp(-1)).tolist()
        first_ref += [t in REF_SET for t in lp.argmax(-1).tolist()]
    return torch.tensor(scores), sum(first_ref) / len(first_ref)

base, base_ref = run()
print(f"intact: score {base.mean():+.2f}, first token is a refusal opener {base_ref:.2f}\n")
print(f"{'condition':>34} | {'score':>6} {'d_score':>8} {'p':>8} | {'refuses':>7}")
ALL = tuple(range(3, len(L)))
conds = [("r-freeze all MLP, no cut", None, ALL, "mlp", True)]
for K in (3, 8, 20):
    conds += [(f"cut K={K}", K, (), "mlp", False),
              (f"cut K={K} + r-freeze MLP repair", K, REPAIR, "mlp", True),
              (f"cut K={K} + r-freeze MLP control", K, CONTROL, "mlp", True),
              (f"cut K={K} + r-freeze ALL MLP", K, ALL, "mlp", True),
              (f"cut K={K} + r-freeze ALL ATTN", K, ALL, "attn", True)]
out = {}
for label, K, fr, which, only_r in conds:
    s, ref = run(K, fr, which, only_r)
    ds = (s - base).mean().item()
    pv = wilcoxon(s.numpy(), base.numpy()).pvalue if ds != 0 else 1.0
    print(f"{label:>34} | {s.mean():>6.2f} {ds:>+8.2f} {pv:>8.1e} | {ref:>7.2f}")
    out[label] = {"score": s.mean().item(), "d_score": ds, "p": float(pv),
                  "first_token_refusal": ref, "per_prompt": s.tolist()}

(config.ARTIFACTS / "block_repair.json").write_text(json.dumps(out, indent = 2))
