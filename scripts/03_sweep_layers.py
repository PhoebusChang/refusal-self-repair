import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import csv
import json

import torch

import config
from src.model_loader import load_model
from src.prompts import load_split
from src.generation import generate, refusal_rate, degenerate_rate, refusal_score
from src.hooks import ablate

model, tokenizer = load_model()
saved = torch.load(config.ARTIFACTS / "directions.pt")
R_hat, positions = saved["R_hat"], saved["positions"]
assert saved["model_id"] == config.MODEL_ID, "directions.pt came from another model"

harmful = load_split("harmful", "val")[:config.SWEEP_N_HARMFUL]
harmless = load_split("harmless", "val")[:config.SWEEP_N_HARMLESS]
n_layers = model.config.num_hidden_layers
lo, hi = (int(f * n_layers) for f in config.LAYER_FRAC)

base = refusal_rate(generate(model, tokenizer, harmful,
                             max_new_tokens = config.SWEEP_MAX_TOKENS))
base_score = sum(refusal_score(model, tokenizer, harmful)) / len(harmful)
print(f"baseline refusal = {base:.2f}  score = {base_score:+.2f}   "
      f"sweeping layers {lo}-{hi - 1}\n")

config.RESULTS.mkdir(parents = True, exist_ok = True)
out_file = open(config.RESULTS / "sweep.csv", "w", newline = "")
writer = csv.DictWriter(out_file, fieldnames = ["layer", "pos", "harmful_refusal", "harmful_score",
                                                "harmless_refusal", "degenerate", "broken"])
writer.writeheader()
raw_file = open(config.RESULTS / "sweep_completions.jsonl", "w", encoding = "utf-8")

rows = []
for layer in range(lo, hi):
    for p_idx, pos in enumerate(positions):
        with ablate(model, R_hat[layer, p_idx]):
            score = sum(refusal_score(model, tokenizer, harmful)) / len(harmful)
            harm = generate(model, tokenizer, harmful,
                            max_new_tokens = config.SWEEP_MAX_TOKENS)
            safe = generate(model, tokenizer, harmless,
                            max_new_tokens = config.SWEEP_MAX_TOKENS)

        harm_rate = refusal_rate(harm)
        safe_rate = refusal_rate(safe)
        degenerate = degenerate_rate(safe)
        broken = safe_rate > 0.15 or degenerate > 0.15

        raw_file.write(json.dumps({"layer": layer, "pos": pos,
                                   "harmful": harm, "harmless": safe}) + "\n")
        raw_file.flush()

        row = {"layer": layer, "pos": pos, "harmful_refusal": harm_rate,
               "harmful_score": round(score, 3),
               "harmless_refusal": safe_rate, "degenerate": degenerate,
               "broken": broken}
        rows.append(row)
        writer.writerow(row)
        out_file.flush()

        print(f"L{layer:02d} pos={pos:3d}  harm={harm_rate:.2f}  "
              f"score={score:+6.2f}  "
              f"safe={safe_rate:.2f}  degen={degenerate:.2f}  "
              f"{'BROKEN' if broken else ''}", flush = True)

out_file.close()
raw_file.close()

usable = [r for r in rows if not r["broken"]]
if not usable:
    raise SystemExit("every config tripped the 'broken' guard")
best = min(usable, key = lambda r: r["harmful_refusal"])
print(f"\nbaseline={base:.2f}   best: L{best['layer']} pos={best['pos']} "
      f"harmful_refusal={best['harmful_refusal']:.2f}")

saved["best_layer"] = best["layer"]
saved["best_pos_idx"] = positions.index(best["pos"])
torch.save(saved, config.ARTIFACTS / "directions.pt")
print("updated", config.ARTIFACTS / "directions.pt")
