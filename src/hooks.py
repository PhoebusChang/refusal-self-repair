from contextlib import contextmanager

import torch

def make_ablation_hook(direction):
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        d = direction.to(h.dtype)
        h = h - (h @ d).unsqueeze(-1) * d
        return (h, *output[1:]) if is_tuple else h

    return hook

def install_ablation(model, direction):
    direction = direction.to(model.device)
    direction = direction / direction.norm()
    hook = make_ablation_hook(direction)

    handles = [model.model.embed_tokens.register_forward_hook(hook)]
    handles += [layer.register_forward_hook(hook) for layer in model.model.layers]
    return handles

def remove_hooks(handles):
    for h in handles:
        h.remove()

@contextmanager
def ablate(model, direction):
    handles = install_ablation(model, direction)
    try:
        yield
    finally:
        remove_hooks(handles)

@contextmanager
def ablate_features(model, banks, keep = None):
    from src.sae import bank_encode

    def make(bank, mask):
        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            z = bank_encode(bank, h.to(torch.float32))
            if mask is not None:
                z = z * mask
            h = h - (z @ bank["W_dec"]).to(h.dtype)
            return (h, *output[1:]) if is_tuple else h
        return hook

    handles = []
    for layer, bank in banks.items():
        mask = None if keep is None else keep[layer].to(bank["W_dec"].dtype)
        handles.append(model.model.layers[layer].register_forward_hook(make(bank, mask)))
    try:
        yield
    finally:
        remove_hooks(handles)
