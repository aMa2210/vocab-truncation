# Prompt experiment: does the instruction change how reasoning survives vocabulary truncation?

The model is only allowed to emit tokens from a shrinking slice of its vocabulary, and
we measure how accuracy falls as the slice shrinks. This folder compares three
instructions given to the model, under two decoders.

## The setup in one paragraph

**Vocabulary truncation.** At every generation step, all tokens outside the allowed set
get logit −∞. The allowed set is the lowest-index fraction **α** of the model's ordinary
tokens (α = 1 keeps everything, α = 0.001 keeps 0.1%). Structural tokens (end-of-turn
etc.) and the characters of the required answer line `Answer: X` are always allowed, so
the model can always finish and write its answer. See `src/truncation.py`.

**Instructions** (`src/prompts.py`; the short names are used in code and file names):

| name | instruction appended to the question |
|---|---|
| `cot` | Think step by step, then give your final answer on a new line in exactly this format: `Answer: <letter/number>` |
| `simple` | Think step by step **using only simple, common words**, then give your final answer … |
| `nocot` | Answer with a single letter/number and nothing else. |

**Decoders.** *Greedy*: always the most likely allowed token (`src/run_eval.py`).
*SMC* (Sequential Monte Carlo, `src/run_smc.py`): 32 sampled continuations ("particles")
whose weights track how much each one is pushed against the vocabulary limit; low-weight
particles are dropped and high-weight ones duplicated. It approximates the model's own
distribution conditioned on staying inside the allowed vocabulary, rather than greedy's
step-by-step local masking. A single-sample run (`n1`) separates "sampling vs greedy"
from "SMC reweighting vs sampling".

**Grid.** Qwen3.5-2B, Qwen3.5-9B, gemma-4-E4B-it (greedy only) × tinyGSM8K, tinyMMLU
(100 questions each, `data/`) × α ∈ {1, 0.1, 0.05, 0.001}.

**Scoring** (`src/parsing.py`): a strict parser reads the `Answer: X` line; if that
fails, an LLM oracle (Gemini, `GEMINI_API_KEY`) may extract the answer, but only if it
can quote the sentence that states it, verbatim, from the model's output. Both scores
are kept (`correct` = with oracle, `tier` says which path produced it).

## Results (bundled)

`figures/prompt_compare.png` – the three instructions under greedy decoding.
`figures/prompt_decoder.png` – `cot` and `simple` under greedy vs SMC.

Main readings (oracle scoring, 100 questions per point, so single points carry ±0.1 noise):

* **Answering directly is flat in α** for the Qwen models (e.g. 9B MMLU 0.74 at every
  α): the answer itself is never truncated, so the collapse of `cot` happens inside the
  reasoning. (gemma-4 MMLU is the exception: 0.50 → 0.63.)
* **"Simple words" recovers much of the collapse at α = 0.1–0.05**, most for the 9B
  model (GSM8K at α = 0.05: 0.37 → 0.77; MMLU: 0.27 → 0.60), and for gemma-4 even at
  α = 0.001 (GSM8K 0.09 → 0.51). It costs nothing at α = 1 for 9B but ~0.1 for 2B, and it
  does not help Qwen at α = 0.001.
* **SMC does not recover it.** Pooled over all 16 cells at α ∈ {0.1, 0.05}, SMC is
  0.065 below greedy (95% CI 0.037–0.092) and 0.033 below single sampling, although it
  stops (emits end-of-sequence) more often than either decoder in every truncated cell.

## Layout

```
src/run_eval.py        greedy runs (all three instructions)
src/prompts.py         the instructions
src/truncation.py      the allowed-token set for a given alpha
src/parsing.py         strict parser + oracle
src/run_smc.py         SMC / single-sample runs (separate environment)
src/smc_compat.py      runtime fix for a bug in genlm-backend 0.3.0, used by run_smc.py
src/aggregate_smc.py   oracle re-scoring of SMC outputs + summary table
src/plot_prompts.py    the two figures
analysis/inflation.json, make_inflation.py
                       per-alpha token-budget scaling (a truncated vocabulary needs more
                       tokens for the same text, so the budget grows with it)
run_prompts.sh         the greedy grid        run_smc_grid.sh   the SMC grid
results/               greedy outputs, one JSONL per cell (full model output per question)
results_smc_oracle/    SMC outputs, oracle-scored, per question WITHOUT the 32 particle
                       texts (the full files are ~180 MB; regenerate with the scripts)
```

File names: `<model>__<task>__a<alpha>__r1[__nocot|__simple][__asciiex][__smc32|__n1].jsonl`.
No suffix means the `cot` instruction; `__asciiex` marks gemma-4 runs (below).

## Reproducing

Two Python 3.11 environments, because the SMC library pins its own transformers/vllm:

```bash
python3.11 -m venv venv     && venv/bin/pip install -r requirements.txt
python3.11 -m venv venv_smc && venv_smc/bin/pip install -r requirements_smc.txt
export GEMINI_API_KEY=...   # or put GEMINI_API_KEY=... in ./api_key.env
```

```bash
./run_prompts.sh                                   # greedy grid, ~1 GPU-day on one 46 GB GPU
./run_smc_grid.sh                                  # SMC grid, ~3 GPU-days
venv/bin/python src/aggregate_smc.py --rescore     # oracle-score SMC outputs (~14k Gemini calls, ~$80)
venv/bin/python src/plot_prompts.py                # figures/
```

Every script skips cells whose output file already exists; `run_smc.py` also resumes
inside a cell. To skip the oracle entirely (free, no API key), pass `--no-oracle` to
`run_eval.py` and skip the `aggregate_smc.py --rescore` step; `run_smc.py` never calls it.

## Things to know

* **gemma-4 needs `--exempt-ascii`.** Its tokenizer keeps the characters it actually
  uses (digits, space, punctuation) near the top of the vocabulary, so a plain index cut
  removes them at α = 0.9. The flag always allows every single-character ASCII token; it
  changes nothing for Qwen, whose alphabet is at the bottom of the vocabulary.
* **gemma-4 has no SMC results**: vllm 0.26 cannot build its model configuration.
* **Greedy runs are not bit-reproducible** on GPU (about 80 of 100 outputs can differ
  between identical runs); SMC runs are, given the per-question seed.
* **`n1` is AWRS sampling, not plain masked sampling**: at a step where the allowed
  tokens hold < 1e-5 of the probability, the sampler discards the sequence (`dead`,
  counted in `n_dead`) where plain masked sampling would renormalise and continue.
* **Oracle caveat**: the oracle model "thinks" before answering and occasionally uses up
  its 2,000-token output limit before writing the answer; the call is then scored as no
  answer. The same configuration was used for every arm.
* **Token budget**: 1,024 new tokens at α = 1, scaled per α by `analysis/inflation.json`
  (e.g. ×2.33 for GSM8K at α = 0.001); 24 tokens for `nocot`.
