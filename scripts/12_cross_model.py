import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json

import torch
from scipy.stats import wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from src.prompts import load_split, to_chat
from src.generation import refusal_rate, degenerate_rate, REFUSAL_OPENERS, COMPLY_OPENERS
from src.repair import run, direction_cut, block, opener_ids, STATE, _split, _join

MODEL = sys.argv[1]
TAG = MODEL.split("/")[-1]
POSITIONS = (-1, -2, -3, -4, -5)
MAXTOK = 64
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


# direction, position and layer
R = position_means(load_split("harmful", "train")) - position_means(load_split("harmless", "train"))
R = R / R.norm(dim = -1, keepdim = True)
val = load_split("harmful", "val")
base_val = mean(run(model, tok, val, ref_ids = REF, com_ids = COM)[0])
best, best_pos, best_s = None, None, float("inf")
for j, pos in enumerate(POSITIONS):
    for l in range(int(0.2 * n), int(0.8 * n) + 1):
        s = mean(run(model, tok, val, lambda: direction_cut(model, R[j, l], range(n)),
                     ref_ids = REF, com_ids = COM)[0])
        if s < best_s:
            best, best_pos, best_s = l, pos, s
r = R[POSITIONS.index(best_pos), best]
g = torch.Generator().manual_seed(config.SEED)
u = torch.randn(r.numel(), generator = g).to(DEV)
u = u - (u @ r) * r
u = u / u.norm()
print(f"{TAG}: {n} layers, refusal direction L{best} pos {best_pos} "
      f"(val score {base_val:+.2f} -> {best_s:+.2f} under full ablation)\n", flush = True)

# partial cuts, with and without the repair
test = load_split("harmful", "test")
TAP, RM = {}, []
tap = lambda h: TAP.setdefault("p", []).append((h @ r).cpu())


def partial_cut(layers, alpha):
    def hook(module, args, output):
        h, rest = _split(output)
        x = h.float()
        proj = (x @ r) * alpha * STATE["content"]
        if STATE["phase"] == "edit" and x.shape[1] > 1:
            RM.append(proj[:, -1].cpu())
        return _join((x - proj[..., None] * r).to(h.dtype), rest)
    return lambda: [L[l].register_forward_hook(hook) for l in layers]


def measure(edits = None, blocks = None):
    TAP.clear()
    s, f, _ = run(model, tok, test, edits, blocks, ref_ids = REF, com_ids = COM, tap = tap)
    refuse = sum(t in set(REF) for t in f) / len(f)
    return torch.tensor(s), torch.cat(TAP["p"]), refuse


s0, p0, ref0 = measure()
chk, _, _ = measure(None, lambda: block(model, range(n), "mlp", r))
print(f"intact: score {s0.mean():+.2f}, first token refuses {ref0:.2f}; "
      f"block-only identical: {torch.equal(chk, s0)}\n")
print(f"{'design':>10} {'removed':>8} {'repair':>7} | {'cut':>13} {'+blk MLP r':>13} "
      f"{'+blk ATTN r':>13} {'+blk MLP u':>13}   (score / first-token refusal)")

band = list(range(max(best - 2, 0), min(best + 3, n)))
designs = {"single25": ([best], 0.25), "single50": ([best], 0.5),
           "single": ([best], 1.0), "band": (band, 1.0),
           "alpha0.3": (list(range(n - 1)), 0.3), "alpha0.5": (list(range(n - 1)), 0.5)}
rows, CUTS = {}, {}
for name, (layers, alpha) in designs.items():
    down = range(layers[0] + 1, n) if len(layers) < n - 1 else range(n)
    cut = partial_cut(layers, alpha)
    CUTS[name] = (cut, down)
    RM.clear()
    res = {"cut": measure(cut)}
    removed = sum(t.sum().item() for t in RM) / len(test)
    res["mlp_r"] = measure(cut, lambda: block(model, down, "mlp", r))
    res["attn_r"] = measure(cut, lambda: block(model, down, "attn", r))
    res["mlp_u"] = measure(cut, lambda: block(model, down, "mlp", u))
    repair = 1 - (p0 - res["cut"][1]).mean().item() / removed
    cell = lambda k: f"{res[k][0].mean():+6.2f} / {res[k][2]:.2f}"
    print(f"{name:>10} {removed:>8.1f} {repair:>7.2f} | {cell('cut'):>13} "
          f"{cell('mlp_r'):>13} {cell('attn_r'):>13} {cell('mlp_u'):>13}", flush = True)
    rows[name] = {"layers": layers, "alpha": alpha, "removed": removed, "repair": repair,
                  **{k: {"score": v[0].mean().item(), "first_token_refusal": v[2],
                         "scores": v[0].tolist()} for k, v in res.items()},
                  "p_block_vs_cut": float(wilcoxon(res["mlp_r"][0], res["cut"][0]).pvalue),
                  "p_block_r_vs_u": float(wilcoxon(res["mlp_r"][0], res["mlp_u"][0]).pvalue)}

# generations for the design where the pin moves first-token refusal most
lg = max(rows, key = lambda k: (rows[k]["cut"]["first_token_refusal"]
                                - rows[k]["mlp_r"]["first_token_refusal"],
                                rows[k]["cut"]["score"] - rows[k]["mlp_r"]["score"]))
cut, down = CUTS[lg]
harmless = load_split("harmless", "test")[:50]
gens = {}
for label, ed, bl in [("intact", None, None),
                      ("direction ablation (all layers)", lambda: direction_cut(model, r, range(n)), None),
                      (f"cut {lg}", cut, None),
                      (f"cut {lg} + block MLP r", cut, lambda: block(model, down, "mlp", r)),
                      (f"cut {lg} + block MLP u (random)", cut, lambda: block(model, down, "mlp", u))]:
    _, _, h = run(model, tok, test, ed, bl, MAXTOK)
    _, _, hl = run(model, tok, harmless, ed, bl, MAXTOK)
    gens[label] = {"asr_keyword": 1 - refusal_rate(h), "degen": degenerate_rate(h),
                   "harmless_refusal": refusal_rate(hl), "harmful": h, "harmless": hl}
    print(f"  gen {label:>34}: ASR(keyword) {gens[label]['asr_keyword']:.2f}  "
          f"degen {gens[label]['degen']:.2f}  harmless refusal "
          f"{gens[label]['harmless_refusal']:.2f}", flush = True)

(config.ARTIFACTS / f"cross_{TAG}.json").write_text(json.dumps(
    {"model": MODEL, "n_layers": n, "direction_layer": best, "direction_pos": best_pos,
     "val_score_intact": base_val, "val_score_full_ablation": best_s,
     "intact_score": s0.mean().item(), "rows": rows, "gen_design": lg,
     "prompts": test, "harmless_prompts": harmless, "conditions": gens}, indent = 2))
