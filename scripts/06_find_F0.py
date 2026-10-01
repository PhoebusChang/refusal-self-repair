# candidate features by cos(W_dec, r)

import json
import torch
import torch.nn.functional as F
import config
from src.sae import load_sae, align

saved = torch.load(config.ARTIFACTS / "directions.pt")
BEST = saved["best_layer"]
r = saved["R_hat"][BEST, saved["best_pos_idx"]]

ranked, ctrl_max = {}, 0.0
for layer in config.SAE_LAYERS:
    sae = load_sae(layer)
    cos = align(sae, r)
    top = cos.topk(config.F0_TOPK)
    ranked[str(layer)] = {"ids": top.indices.tolist(),
                          "cos": [round(v, 4) for v in top.values.tolist()]}
    rc = F.cosine_similarity(torch.randn_like(sae.W_dec),
                             (r / r.norm()).to(cos.device).unsqueeze(0), dim = 1)
    ctrl_max = max(ctrl_max, rc.max().item())
    print(f"  L{layer:02d}  top cos {top.values[0]:.3f}   "
          f"#>0.15 {(cos > 0.15).sum().item():4d}   K-th {top.values[-1]:.3f}")
    del sae
    torch.cuda.empty_cache()

out = {"direction_layer": BEST, "pos_idx": saved["best_pos_idx"],
       "layers": list(config.SAE_LAYERS), "topk": config.F0_TOPK,
       "ranked": ranked, "control_max_cos": round(ctrl_max, 4)}
(config.ARTIFACTS / "F0.json").write_text(json.dumps(out))
tot = len(config.SAE_LAYERS) * config.F0_TOPK
print(f"\ncandidates: {config.F0_TOPK}/layer x {len(config.SAE_LAYERS)} layers = {tot}")
print(f"random-decoder control max cos: {ctrl_max:.3f}")
