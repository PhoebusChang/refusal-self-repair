from pathlib import Path
import torch

# paths
ROOT = Path(__file__).parent
DATA = ROOT / "data"
SPLITS = DATA / "splits"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

# model
MODEL_ID = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_IN_4BIT = False
ATTN_IMPL = "eager"

# sae
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_WIDTH = "16k"

SAE_LAYERS = tuple(range(3, 26))
F0_TOPK = 200

# data
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
ALPACA_ID = "tatsu-lab/alpaca"

N_TRAIN, N_VAL, N_TEST = 128, 32, 100
SEED = 0

# activations
POSITIONS = (-1, -2, -3, -4, -5)
BATCH_SIZE = 4
MAX_NEW_TOKENS = 48
LAYER_FRAC = (0.2, 0.85)

# sweep
SWEEP_N_HARMFUL = 16
SWEEP_N_HARMLESS = 8
SWEEP_MAX_TOKENS = 24
