# ablation split by token position

import json

import torch

import config
from src.model_loader import load_model
from src.prompts import load_split, to_chat
from src.generation import (_opener_ids, REFUSAL_OPENERS, COMPLY_OPENERS,
                            refusal_rate, degenerate_rate)
from src.sae import load_sae, build_bank

KS = (3, 8, 40, 200)
N, MAXTOK = 100, 24
DEV = config.DEVICE

F0 = json.loads((config.ARTIFACTS / "F0.json").read_text())
layers = [l for l in F0["layers"] if str(l) in F0["ranked"]]
saved = torch.load(config.ARTIFACTS / "directions.pt")
r = saved["R_hat"][saved["best_layer"], saved["best_pos_idx"]].to(DEV, torch.float32)
r = r / r.norm()

model, tokenizer = load_model()
harmful = load_split("harmful", "test")[:N]
harmless = load_split("harmless", "test")[:32]

banks = {}
for l in layers:
    sae = load_sae(l)
    banks[l] = build_bank(sae, F0["ranked"][str(l)]["ids"])
    del sae
    torch.cuda.empty_cache()

POS = {}

def where(h, which):
    if h.shape[1] != POS["content"].shape[1]:
        return torch.full(h.shape[:2], which != "bos", device = h.device)
    return POS[which]

def feature_hook(l, K, which):
    b = banks[l]
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        x = h.to(torch.float32)
        pre = x @ b["W_enc"][:, :K] + b["b_enc"][:K]
        z = pre * (pre > b["thresh"][:K])
        x = x - (z @ b["W_dec"][:K]) * where(h, which)[..., None]
        h = x.to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook

def direction_hook(which):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        x = h.to(torch.float32)
        x = x - (x @ r)[..., None] * r * where(h, which)[..., None]
        h = x.to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook

def install(kind, which, K = None):
    if kind == "none":
        return []
    if kind == "feat":
        return [model.model.layers[l].register_forward_hook(feature_hook(l, K, which))
                for l in layers]
    mods = [model.model.embed_tokens] + list(model.model.layers)
    return [m.register_forward_hook(direction_hook(which)) for m in mods]

REF, COM = _opener_ids(tokenizer, REFUSAL_OPENERS), _opener_ids(tokenizer, COMPLY_OPENERS)

@torch.no_grad()
def run(prompts, hooks_fn):
    scores, texts = [], []
    for i in range(0, len(prompts), config.BATCH_SIZE):
        enc = tokenizer([to_chat(tokenizer, p) for p in prompts[i:i + config.BATCH_SIZE]],
                        return_tensors = "pt", padding = True,
                        add_special_tokens = False).to(DEV)
        am = enc["attention_mask"].bool()
        bos = torch.zeros_like(am)
        bos[torch.arange(am.shape[0]), am.float().argmax(1)] = True
        POS["content"], POS["bos"], POS["all"] = am & ~bos, bos, am
        hs = hooks_fn()
        try:
            lp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
            scores += (lp[:, REF].logsumexp(-1) - lp[:, COM].logsumexp(-1)).tolist()
            gen = model.generate(**enc, max_new_tokens = MAXTOK, do_sample = False,
                                 pad_token_id = tokenizer.pad_token_id)
            texts += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                            skip_special_tokens = True)
        finally:
            for h in hs:
                h.remove()
    return scores, texts

mean = lambda v: sum(v) / len(v)
conds = [("none", "all", None), ("dir", "all", None), ("dir", "content", None),
         ("dir", "bos", None)]
conds += [("feat", w, K) for K in KS for w in ("all", "content", "bos")]

print(f"{'ablation':>14} {'where':>8} | {'score':>6} {'ASR':>5} {'degen':>6} | "
      f"{'harmless_ref':>12}")
rows = []
for kind, which, K in conds:
    fn = lambda: install(kind, which, K)
    s, t = run(harmful, fn)
    _, ts = run(harmless, fn)
    row = {"kind": kind, "where": which, "K": K, "score": mean(s),
           "asr": 1 - refusal_rate(t), "degen": degenerate_rate(t),
           "harmless_refusal": refusal_rate(ts), "examples": t[:5]}
    rows.append(row)
    label = {"none": "none", "dir": "direction"}.get(kind, f"feat K={K}")
    print(f"{label:>14} {which:>8} | {row['score']:>6.2f} {row['asr']:>5.2f} "
          f"{row['degen']:>6.2f} | {row['harmless_refusal']:>12.2f}")

(config.ARTIFACTS / "bos_check.json").write_text(json.dumps(rows, indent = 2))
