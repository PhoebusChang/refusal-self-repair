import torch

from src.prompts import to_chat

STATE = {"content": None, "cache": {}, "phase": None}


def is_gemma(layer):
    return hasattr(layer, "pre_feedforward_layernorm")


def writes(layer, part):
    if part == "mlp":
        return layer.post_feedforward_layernorm if is_gemma(layer) else layer.mlp
    return layer.post_attention_layernorm if is_gemma(layer) else layer.self_attn


def _split(output):
    return (output[0], output[1:]) if isinstance(output, tuple) else (output, None)


def _join(h, rest):
    return (h, *rest) if rest is not None else h


def feature_cut(model, banks, K):
    def make(b):
        def hook(module, args, output):
            h, rest = _split(output)
            x = h.float()
            pre = x @ b["W_enc"][:, :K] + b["b_enc"][:K]
            z = pre * (pre > b["thresh"][:K])
            x = x - (z @ b["W_dec"][:K]) * STATE["content"][..., None]
            return _join(x.to(h.dtype), rest)
        return hook
    L = model.model.layers
    return [L[l].register_forward_hook(make(b)) for l, b in banks.items()]


def direction_cut(model, d, layers):
    def hook(module, args, output):
        h, rest = _split(output)
        x = h.float()
        x = x - (x @ d)[..., None] * d * STATE["content"][..., None]
        return _join(x.to(h.dtype), rest)
    L = model.model.layers
    return [L[l].register_forward_hook(hook) for l in layers]


def block(model, layers, part, d):
    def make(key):
        def hook(module, args, output):
            h, rest = _split(output)
            if STATE["phase"] == "intact":
                STATE["cache"][key] = h.detach()
                return output
            diff = ((h.float() - STATE["cache"][key].float()) @ d) * STATE["content"]
            return _join((h.float() - diff[..., None] * d).to(h.dtype), rest)
        return hook
    L = model.model.layers
    return [writes(L[l], part).register_forward_hook(make((part, l))) for l in layers]


def opener_ids(tokenizer, words):
    return sorted({tokenizer.encode(w, add_special_tokens = False)[0] for w in words})


@torch.no_grad()
def run(model, tokenizer, prompts, edits = None, blocks = None, max_new = 0,
        batch = 4, ref_ids = None, com_ids = None, tap = None):
    dev = model.device
    stop = {tokenizer.eos_token_id}
    for t in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tokenizer.unk_token_id:
            stop.add(i)
    scores, firsts, texts = [], [], []

    def fwd(ids, am, cache, phase):
        STATE["phase"] = phase
        pos = (am.cumsum(-1) - 1).clamp(min = 0)[:, -ids.shape[1]:]
        hs = []
        if phase == "edit" and edits:
            hs += edits()
        if blocks:
            hs += blocks()
        try:
            return model(input_ids = ids, attention_mask = am, position_ids = pos,
                         past_key_values = cache, use_cache = True, logits_to_keep = 1)
        finally:
            for h in hs:
                h.remove()

    for i in range(0, len(prompts), batch):
        enc = tokenizer([to_chat(tokenizer, p) for p in prompts[i:i + batch]],
                        return_tensors = "pt", padding = True,
                        add_special_tokens = False).to(dev)
        ids, am = enc["input_ids"], enc["attention_mask"]
        content = am.bool().clone()
        content[torch.arange(ids.shape[0]), am.float().argmax(1)] = False   # BOS
        STATE["content"], STATE["cache"] = content, {}

        cache_i = fwd(ids, am, None, "intact").past_key_values if blocks else None
        if tap:
            hsl = model.model.layers[-1].register_forward_hook(
                lambda m, a, o: tap(_split(o)[0][:, -1].float()))
        try:
            out = fwd(ids, am, None, "edit")
        finally:
            if tap:
                hsl.remove()
        cache_c = out.past_key_values
        lp = torch.log_softmax(out.logits[:, -1].float(), -1)
        if ref_ids is not None:
            scores += (lp[:, ref_ids].logsumexp(-1) - lp[:, com_ids].logsumexp(-1)).tolist()
        nxt = lp.argmax(-1)
        firsts += nxt.tolist()

        new, done = [], torch.zeros(ids.shape[0], dtype = torch.bool, device = dev)
        for _ in range(max_new):
            new.append(nxt)
            done |= torch.tensor([t in stop for t in nxt.tolist()], device = dev)
            if done.all():
                break
            am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
            STATE["content"] = torch.ones_like(am[:, -1:], dtype = torch.bool)
            step = nxt[:, None]
            if blocks:
                STATE["cache"] = {}
                cache_i = fwd(step, am, cache_i, "intact").past_key_values
            out = fwd(step, am, cache_c, "edit")
            cache_c = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        if max_new:
            seqs = torch.stack(new, 1).tolist()
            for s in seqs:
                cut = next((k for k, t in enumerate(s) if t in stop), len(s))
                texts.append(tokenizer.decode(s[:cut], skip_special_tokens = True))
        del cache_i, cache_c
    return scores, firsts, texts
