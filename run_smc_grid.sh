#!/usr/bin/env bash
# SMC arm of the prompt experiment: the same cells as run_prompts.sh for the Qwen models,
# decoded with genlm-control's Sequential Monte Carlo instead of greedy.
#
# Each cell is decoded twice: 32-particle SMC (__smc32) and a single sample (__n1), the
# latter separating "sampling vs greedy" from "SMC reweighting vs sampling". Runs in the
# separate venv_smc environment (see README). Each model is loaded once for all three
# instructions. Every question is checkpointed as it finishes (<cell>.partial) and
# finished cells are skipped, so after an interruption simply re-run this script; a
# resumed cell is byte-identical to an uninterrupted one because question i always uses
# seed0+i.
#
# gemma-4 is not included: vllm 0.26 cannot build its model config (per-layer head_dim).
#
# Then, in the main environment:  venv/bin/python src/aggregate_smc.py --rescore
#
#   setsid nohup ./run_smc_grid.sh > logs/smc_grid.log 2>&1 < /dev/null &
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p results_smc logs

failed=0
for M in Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B; do
  echo "########## $M"
  venv_smc/bin/python src/run_smc.py --model "$M" --prompts nocot cot simple \
      --tasks gsm8k mmlu --particles 32 1 --out results_smc || failed=$((failed+1))
done
echo "SMC_GRID_DONE failed_models=$failed"
exit $((failed > 0))
