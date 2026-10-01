# same, full generations + capability

import json
import random

import torch

import config
from src.model_loader import load_model
from src.prompts import load_split
from src.generation import (refusal_rate, degenerate_rate, REFUSAL_OPENERS,
                            COMPLY_OPENERS)
from src.sae import load_sae, build_bank
from src.repair import run, feature_cut, direction_cut, block, opener_ids

REPAIR = (9, 11, 12, 14, 16, 17)
CONTROL = (19, 20, 21, 22, 23, 24)
MAXTOK = 64
DEV = config.DEVICE

F0 = json.loads((config.ARTIFACTS / "F0.json").read_text())
sae_layers = [l for l in F0["layers"] if str(l) in F0["ranked"]]
saved = torch.load(config.ARTIFACTS / "directions.pt")
r = saved["R_hat"][saved["best_layer"], saved["best_pos_idx"]].to(DEV, torch.float32)
r = r / r.norm()
g = torch.Generator().manual_seed(config.SEED)
u = torch.randn(r.numel(), generator = g).to(DEV)
u = u - (u @ r) * r
u = u / u.norm()

model, tokenizer = load_model()
n = len(model.model.layers)
ALL = tuple(range(3, n))
harmful = load_split("harmful", "test")
harmless = load_split("harmless", "test")[:50]
rng = random.Random(config.SEED)
sums = [(rng.randint(10, 99), rng.randint(10, 99)) for _ in range(40)]
arith = [f"What is {a} + {b}? Reply with only the number." for a, b in sums]
REF, COM = opener_ids(tokenizer, REFUSAL_OPENERS), opener_ids(tokenizer, COMPLY_OPENERS)

banks = {}
for l in sae_layers:
    sae = load_sae(l)
    banks[l] = build_bank(sae, F0["ranked"][str(l)]["ids"][:8])
    del sae
    torch.cuda.empty_cache()

cut = lambda K: (lambda: feature_cut(model, {l: banks[l] for l in sae_layers}, K))
blk = lambda layers, d: (lambda: block(model, layers, "mlp", d))
conds = {
    "intact": (None, None),
    "direction ablation (all layers)": (lambda: direction_cut(model, r, range(n)), None),
    "block MLP r, no cut": (None, blk(ALL, r)),
    "cut K=3": (cut(3), None),
    "cut K=3 + block MLP r": (cut(3), blk(ALL, r)),
    "cut K=3 + block MLP u (random)": (cut(3), blk(ALL, u)),
    "cut K=3 + block 6 repair MLPs": (cut(3), blk(REPAIR, r)),
    "cut K=3 + block 6 control MLPs": (cut(3), blk(CONTROL, r)),
    "cut K=8": (cut(8), None),
    "cut K=8 + block MLP r": (cut(8), blk(ALL, r)),
    "cut K=8 + block MLP u (random)": (cut(8), blk(ALL, u)),
}

print(f"{'condition':>34} | {'score':>6} {'ASR':>5} {'degen':>6} | "
      f"{'harmless_ref':>12} {'arith':>6}", flush = True)
out = {}
for label, (ed, bl) in conds.items():
    s, _, h = run(model, tokenizer, harmful, ed, bl, MAXTOK, ref_ids = REF, com_ids = COM)
    _, _, hl = run(model, tokenizer, harmless, ed, bl, MAXTOK)
    _, _, a = run(model, tokenizer, arith, ed, bl, 8)
    row = {"score": sum(s) / len(s), "asr_keyword": 1 - refusal_rate(h),
           "degen": degenerate_rate(h), "harmless_refusal": refusal_rate(hl),
           "harmless_degen": degenerate_rate(hl),
           "arith": sum(str(x + y) in o for (x, y), o in zip(sums, a)) / len(sums),
           "scores": s, "harmful": h, "harmless": hl}
    out[label] = row
    print(f"{label:>34} | {row['score']:>6.2f} {row['asr_keyword']:>5.2f} "
          f"{row['degen']:>6.2f} | {row['harmless_refusal']:>12.2f} {row['arith']:>6.2f}",
          flush = True)

(config.ARTIFACTS / "gen_gemma.json").write_text(json.dumps(
    {"model": config.MODEL_ID, "prompts": harmful, "harmless_prompts": harmless,
     "conditions": out}, indent = 2))
