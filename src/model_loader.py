import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config

def load_model():
    kwargs = {"device_map": config.DEVICE, "dtype": torch.bfloat16}
    if getattr(config, "ATTN_IMPL", None):
        kwargs["attn_implementation"] = config.ATTN_IMPL      # Gemma-2 softcapping
    if config.LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, **kwargs)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer
