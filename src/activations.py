import torch

import config
from src.prompts import to_chat

@torch.no_grad()
def get_resid(model, tokenizer, prompts, positions=None, batch_size=None):
    positions = list(positions if positions is not None else config.POSITIONS)
    batch_size = batch_size or config.BATCH_SIZE

    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [to_chat(tokenizer, p) for p in prompts[i:i + batch_size]]
        enc = tokenizer(texts, return_tensors = "pt", padding = True,
                        add_special_tokens = False).to(model.device)

        hidden = model(**enc, output_hidden_states = True).hidden_states
        stacked = torch.stack(hidden, dim = 0)
        out.append(stacked[:, :, positions, :].float().cpu())
        del hidden, stacked

    return torch.cat(out, dim = 1)
