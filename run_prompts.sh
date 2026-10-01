#!/usr/bin/env bash
# Greedy arm of the prompt experiment: three instructions x three models x two tasks x
# the alphas in src/prompts.py (ALPHAS), 100 questions per cell, greedy decoding.
#
#   cot     "Think step by step, then give your final answer ..."
#   simple  "Think step by step using only simple, common words, then ..."
#   nocot   "Answer with a single letter/number and nothing else."
#
# Writes results/<model>__<task>__a<alpha>__r1[__simple|__nocot][__asciiex].jsonl. Each
# model is loaded once for all three instructions; cells whose file already exists are
# skipped, so the script is safe to re-run. The model list mirrors prompts.MODELS.
#
# gemma-4 needs --exempt-ascii: its tokenizer keeps the characters it actually encodes
# with (digits, space, punctuation) at the 92.5th percentile of the vocabulary, so even
# alpha=0.9 would delete them. The flag keeps every single-character ASCII token and is
# a no-op for Qwen, whose alphabet already sits at the bottom of the vocabulary.
#
#   ./run_prompts.sh > logs/prompts.log 2>&1
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p results logs

failed=0
run() {  # run <model> <extra flags...>
  echo "########## $*"
  venv/bin/python src/run_eval.py --model "$@" --tasks gsm8k mmlu \
      --prompts cot simple nocot --batch-size 25 --rep 1 --out results || failed=$((failed+1))
}

run Qwen/Qwen3.5-2B
run Qwen/Qwen3.5-9B
run google/gemma-4-E4B-it --exempt-ascii

echo "PROMPTS_DONE failed_models=$failed"
exit $((failed > 0))
