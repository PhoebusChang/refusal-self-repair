# Gate B: SAE reconstruction quality

import json
import torch
import config
from src.model_loader import load_model
from src.prompts import load_split, to_chat
from src.sae import load_sae, fvu

saved = torch.load(config.ARTIFACTS / "directions.pt")
LAYER = saved["best_layer"]

model, tokenizer = load_model()
prompts = load_split("harmful", "val")[:16] + load_split("harmless", "val")[:16]
enc = tokenizer([to_chat(tokenizer, p) for p in prompts], return_tensors = "pt",
                padding = True, add_special_tokens = False).to(model.device)
with torch.no_grad():
    hs = model(**enc, output_hidden_states = True).hidden_states[LAYER + 1]

mask = enc["attention_mask"].clone()
for r in range(mask.shape[0]):
    mask[r, mask[r].nonzero()[0].item()] = 0        # drop BOS (attention sink)

sae = load_sae(LAYER)
f_content, l0 = fvu(sae, hs[mask.bool()])
f_tok, _ = fvu(sae, hs[:, -1, :])

out = {"layer": LAYER, "fvu": round(f_content, 4), "explained_var": round(1 - f_content, 4),
       "l0": round(l0, 1), "fvu_single_token_artifact": round(f_tok, 4),
       "passed": f_content < 0.35}
(config.RESULTS / "gate_b.json").write_text(json.dumps(out, indent = 2))
print(f"layer {LAYER}  FVU {f_content:.3f}  explained var {1-f_content:.3f}  L0 {l0:.0f}")
print(f"(single-token measurement would report FVU {f_tok:.3f} - artifact, not a real failure)")
print(f"\nGATE B: {'PASS' if out['passed'] else 'FAIL'}")
