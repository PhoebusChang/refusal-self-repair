import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import config
from src.model_loader import load_model
from src.prompts import load_split
from src.activations import get_resid

model, tokenizer = load_model()

harmful = load_split("harmful", "train")
harmless = load_split("harmless", "train")
print(f"harmful={len(harmful)}  harmless={len(harmless)}")

H = get_resid(model, tokenizer, harmful)
S = get_resid(model, tokenizer, harmless)
print("shapes:", tuple(H.shape), tuple(S.shape))

R = H.mean(dim = 1) - S.mean(dim = 1)
R_hat = R / R.norm(dim = -1, keepdim = True)

norms = R.norm(dim = -1)
print("\nnorm by layer x position", config.POSITIONS)
for L in range(R.shape[0]):
    print(f"L{L:02d} " + " ".join(f"{n:7.2f}" for n in norms[L]))

print("\ncos(pos -1, pos -2), every 4th layer")
print([round(F.cosine_similarity(R_hat[L, 0], R_hat[L, 1], dim = 0).item(), 2)
       for L in range(0, R.shape[0], 4)])

config.ARTIFACTS.mkdir(parents = True, exist_ok = True)
torch.save({
    "R_hat": R_hat,
    "norms": norms,
    "positions": list(config.POSITIONS),
    "model_id": config.MODEL_ID,
    "n_train": len(harmful),
    "load_in_4bit": config.LOAD_IN_4BIT,
}, config.ARTIFACTS / "directions.pt")
print("\nsaved", config.ARTIFACTS / "directions.pt")
