import torch
import torch.nn.functional as F
from sae_lens import SAE

import config

def load_sae(layer: int, device = None):
    device = device or config.DEVICE
    sae = SAE.from_pretrained(
        config.SAE_RELEASE, f"layer_{layer}/width_{config.SAE_WIDTH}/canonical",
        device = device,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    return sae.to(device).to(torch.float32).eval()

@torch.no_grad()
def fvu(sae, acts):
    x = acts.to(config.DEVICE, torch.float32)
    z = sae.encode(x)
    xh = sae.decode(z)
    f = ((x - xh).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
    return f, (z > 0).float().sum(-1).mean().item()

@torch.no_grad()
def align(sae, direction):
    d = direction.to(config.DEVICE, torch.float32)
    d = d / d.norm()
    return F.cosine_similarity(sae.W_dec.to(torch.float32), d.unsqueeze(0), dim = 1)

@torch.no_grad()
def build_bank(sae, ids):
    ids = torch.as_tensor(list(ids), device = sae.W_enc.device, dtype = torch.long)
    return {
        "ids": ids,
        "W_enc": sae.W_enc[:, ids].contiguous(),          # (d_model, k)
        "b_enc": sae.b_enc[ids].contiguous(),             # (k,)
        "thresh": sae.threshold[ids].contiguous(),        # (k,)
        "W_dec": sae.W_dec[ids].contiguous(),             # (k, d_model)
    }

@torch.no_grad()
def bank_encode(bank, x):
    pre = x @ bank["W_enc"] + bank["b_enc"]
    return pre * (pre > bank["thresh"])
