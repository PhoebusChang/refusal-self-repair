import json

import config

def to_chat(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize = False,
        add_generation_prompt = True
    )

def load_split(name: str, split: str) -> list[str]:
    return json.loads((config.SPLITS / f"{name}.json").read_text(encoding = "utf-8"))[split]
