# AdvBench + Alpaca -> train/val/test

import json
import random

import pandas as pd
from datasets import load_dataset

import config

random.seed(config.SEED)

harmful = pd.read_csv(config.ADVBENCH_URL)["goal"].tolist()
harmless = [r["instruction"] for r in load_dataset(config.ALPACA_ID, split = "train")
            if r["input"].strip() == ""]

harmful = sorted(set(h.strip() for h in harmful))
harmless = sorted(set(h.strip() for h in harmless))
print(f"harmful={len(harmful)}  harmless={len(harmless)}")

def make_split(items, n_train, n_val, n_test):
    items = list(items)
    random.shuffle(items)
    need = n_train + n_val + n_test
    assert len(items) >= need, f"need {need}, have {len(items)}"
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:need],
    }

splits = {
    "harmful": make_split(harmful, config.N_TRAIN, config.N_VAL, config.N_TEST),
    "harmless": make_split(harmless, config.N_TRAIN, config.N_VAL, config.N_TEST),
}

config.SPLITS.mkdir(parents = True, exist_ok = True)
for name, d in splits.items():
    path = config.SPLITS / f"{name}.json"
    path.write_text(json.dumps(d, indent = 2), encoding = "utf-8")
    for k, v in d.items():
        words = sum(len(x.split()) for x in v) / len(v)
        print(f"{name:9s} {k:5s} n={len(v):4d}  mean_words={words:.1f}")
    print(f"  example: {d['train'][0][:70]}")
