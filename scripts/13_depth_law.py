import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from src.prompts import load_split, to_chat
from src.generation import REFUSAL_OPENERS, COMPLY_OPENERS
from src.repair import run, direction_cut, opener_ids, STATE, _split, _join

MODEL = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_ID
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TAG = MODEL.split("/")[-1]
POSITIONS = (-1, -2, -3, -4, -5)
DEV = config.DEVICE

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype = torch.bfloat16, device_map = DEV,
    attn_implementation = "eager" if "gemma" in MODEL.lower() else "sdpa").eval()
model.requires_grad_(False)
L = model.model.layers
n = len(L)
REF, COM = opener_ids(tok, REFUSAL_OPENERS), opener_ids(tok, COMPLY_OPENERS)
mean = lambda v: sum(v) / len(v)


@torch.no_grad()
def position_means(prompts):
    acc = torch.zeros(len(POSITIONS), n, model.config.hidden_size, device = DEV)
    for i in range(0, len(prompts), 8):
        enc = tok([to_chat(tok, p) for p in prompts[i:i + 8]], return_tensors = "pt",
                  padding = True, add_special_tokens = False).to(DEV)
        hs = model(**enc, output_hidden_states = True, logits_to_keep = 1).hidden_states
        for j, pos in enumerate(POSITIONS):
            acc[j] += torch.stack([h[:, pos].float().sum(0) for h in hs[1:]])
    return acc / len(prompts)


R = position_means(load_split("harmful", "train")) - position_means(load_split("harmless", "train"))
R = R / R.norm(dim = -1, keepdim = True)
val = load_split("harmful", "val")
best, best_pos, best_s = None, None, float("inf")
for j, pos in enumerate(POSITIONS):
    for l in range(int(0.2 * n), int(0.8 * n) + 1):
        s = mean(run(model, tok, val, lambda: direction_cut(model, R[j, l], range(n)),
                     ref_ids = REF, com_ids = COM)[0])
        if s < best_s:
            best, best_pos, best_s = l, pos, s
r = R[POSITIONS.index(best_pos), best]

test = load_split("harmful", "test")
TAP, RM = {}, []
tap = lambda h: TAP.setdefault("p", []).append((h @ r).cpu())


def cut_at(layer):
    def hook(module, args, output):
        h, rest = _split(output)
        x = h.float()
        proj = (x @ r) * ALPHA * STATE["content"]
        if STATE["phase"] == "edit" and x.shape[1] > 1:
            RM.append(proj[:, -1].cpu())
        return _join((x - proj[..., None] * r).to(h.dtype), rest)
    return lambda: [L[layer].register_forward_hook(hook)]


def measure(edits = None):
    TAP.clear()
    RM.clear()
    s, _, _ = run(model, tok, test, edits, None, ref_ids = REF, com_ids = COM, tap = tap)
    removed = sum(t.sum().item() for t in RM) / len(test) if RM else 0.0
    return torch.tensor(s), torch.cat(TAP["p"]), removed


s0, p0, _ = measure()
print(f"{TAG}: {n} layers, r from L{best} pos {best_pos}, alpha={ALPHA}")
print(f"intact: score {s0.mean():+.2f}, final projection {p0.mean():.1f}\n")
print(f"{'layer':>6} {'depth':>6} {'after':>6} {'removed':>8} {'missing':>8} "
      f"{'repair':>7} {'effect':>7} {'efficiency':>11}")

rows = []
for layer in range(1, n):
    s, p, removed = measure(cut_at(layer))
    if removed < 1e-6:
        continue
    missing = (p0 - p).mean().item()
    effect = (s0 - s).mean().item()
    rows.append({"layer": layer, "depth": layer / n, "layers_after": n - 1 - layer,
                 "removed": removed, "missing": missing,
                 "repair": 1 - missing / removed,
                 "effect": effect, "efficiency": effect / removed,
                 "score": s.mean().item()})
    d = rows[-1]
    print(f"{layer:>6} {d['depth']:>6.2f} {d['layers_after']:>6} {removed:>8.2f} "
          f"{missing:>8.2f} {d['repair']:>7.2f} {effect:>7.2f} {d['efficiency']:>11.4f}",
          flush = True)


import numpy as np
dep = np.array([d["depth"] for d in rows])
rep = np.array([d["repair"] for d in rows])
eff = np.array([d["efficiency"] for d in rows])
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
bestL = max(rows, key = lambda d: d["efficiency"])
print(f"\ncorr(depth, repair)     = {corr(dep, rep):+.2f}   "
      f"(negative = later edits are repaired less)")
print(f"corr(depth, efficiency) = {corr(dep, eff):+.2f}   "
      f"(positive = later edits buy more behaviour per unit damage)")
print(f"corr(repair, efficiency)= {corr(rep, eff):+.2f}")
print(f"most efficient layer: L{bestL['layer']} (depth {bestL['depth']:.2f}), "
      f"efficiency {bestL['efficiency']:.4f}, repair {bestL['repair']:.2f}")

(config.ARTIFACTS / f"depth_{TAG}.json").write_text(json.dumps(
    {"model": MODEL, "n_layers": n, "alpha": ALPHA, "direction_layer": best,
     "direction_pos": best_pos, "intact_score": s0.mean().item(),
     "intact_projection": p0.mean().item(), "rows": rows,
     "corr_depth_repair": corr(dep, rep), "corr_depth_efficiency": corr(dep, eff),
     "corr_repair_efficiency": corr(rep, eff), "best_layer": bestL["layer"]}, indent = 2))
