#!/bin/bash
#   ./run.sh          01-16 on gemma-2-2b-it
#   ./run.sh cross    also 12-13 on the other three model families
cd "$(dirname "$0")" || exit 1
if [ -z "$PY" ]; then
  for c in ".venv/Scripts/python.exe" ".venv/bin/python" \
           "/e/Research Paper/AI Ethics/.venv/Scripts/python.exe"; do
    [ -x "$c" ] && PY="$c" && break
  done
fi
PY="${PY:-python}"
"$PY" -c "import torch, transformers, sae_lens" 2>/dev/null \
  || { echo "interpreter '$PY' is missing deps; pip install -r requirements.txt"; exit 1; }
echo "python: $PY"
export HF_HOME="${HF_HOME:-D:/huggingface}"
export PYTHONUNBUFFERED=1
mkdir -p results artifacts
echo "repo: $(pwd)"

run () {
  echo ""; echo "########## $1 $2   $(date +%H:%M:%S)"
  "$PY" "scripts/$1.py" $2 2>&1 | tr '\r' '\n' \
    | grep -vE "Loading weights|Downloading|Generating|^[[:space:]]*$" | tail -16
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "!!!! FAILED at $1"; exit 1; fi
}

for s in 01_build_datasets 02_compute_direction 03_sweep_layers 04_gate_a \
         05_gate_b 06_find_F0 07_bos_check 08_repair_fraction \
         09_repair_specificity 10_block_repair 11_full_generation; do
  run "$s"
done

run 13_depth_law google/gemma-2-2b-it

if [ "$1" = "cross" ]; then
  for m in meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
           Qwen/Qwen2.5-3B-Instruct; do
    run 12_cross_model "$m"
    run 13_depth_law "$m"
  done
fi

run 14_judge
run 15_stats
run 16_figures

echo ""; echo "########## COMPLETE $(date +%H:%M:%S)"
