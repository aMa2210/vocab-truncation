"""Questions, instructions and result-file names: the single definition of each.

Every model here is an instruct model prompted through its chat template with
enable_thinking=False (no hidden reasoning trace). Both runners (run_eval.py, run_smc.py),
the aggregator and the plotter import from this module, so an instruction, a filename
suffix or a token budget cannot differ between them.
"""
import pyarrow.parquet as pq

TASK_FILES = {"gsm8k": "data/tinyGSM8k_test.parquet",
              "mmlu": "data/tinyMMLU_test.parquet"}

# The grid. Runners default to these alphas; the aggregator and plotter read both lists.
ALPHAS = (1, 0.1, 0.05, 0.001)
# (Hugging Face repo, display name, exempt_ascii, has SMC results). Keep in step with the
# model loops in run_prompts.sh / run_smc_grid.sh.
MODELS = (("Qwen/Qwen3.5-2B", "Qwen3.5-2B", False, True),
          ("Qwen/Qwen3.5-9B", "Qwen3.5-9B", False, True),
          ("google/gemma-4-E4B-it", "Gemma-4-E4B", True, False))
LETTERS = "ABCDEFGH"

# The three instructions. `suffix` goes into the result filename, `budget` overrides the
# alpha-scaled token budget (None = scaled), `reasons` says whether the model is asked to
# write a reasoning trace.
#
# nocot is the control: the prompt is never masked and an MCQ answer is a single
# low-index token, so a model can answer without using the truncated vocabulary at all.
# Measuring it at each alpha separates "the reasoning survived truncation" from "the
# answer channel was never truncated in the first place".
#
# simple has the same contract as cot, with the constraint moved into the instruction:
# the model is asked to keep to simple words instead of having the vocabulary cut out from
# under it. Comparing it against cot separates "a restricted vocabulary is fatal" from "an
# externally imposed hard mask is fatal".
_FORMAT = " on a new line in exactly this format:\nAnswer: <{}>"
VARIANTS = {
    "cot":    dict(mcq="Think step by step, then give your final answer" + _FORMAT.format("letter"),
                   num="Think step by step, then give your final answer" + _FORMAT.format("number"),
                   suffix="", budget=None, reasons=True),
    "simple": dict(mcq="Think step by step using only simple, common words, then give your "
                       "final answer" + _FORMAT.format("letter"),
                   num="Think step by step using only simple, common words, then give your "
                       "final answer" + _FORMAT.format("number"),
                   suffix="__simple", budget=None, reasons=True),
    "nocot":  dict(mcq="Answer with a single letter and nothing else.\nAnswer:",
                   num="Answer with a single number and nothing else.\nAnswer:",
                   suffix="__nocot", budget=24, reasons=False),
}


def load_rows(task, limit=0):
    rows = pq.read_table(TASK_FILES[task]).to_pylist()
    return rows[:limit] if limit else rows


def build(row, task, variant):
    """-> (chat messages, gold answer, kind) with kind in {"mcq", "num"}."""
    if task == "mmlu":
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        body, gold, kind = f"{row['question']}\n{opts}", LETTERS[int(row["answer"])], "mcq"
    elif task == "gsm8k":
        body = row["question"]
        gold, kind = row["answer"].split("####")[-1].strip().replace(",", ""), "num"
    else:
        raise ValueError(task)
    return [{"role": "user", "content": f"{body}\n\n{VARIANTS[variant][kind]}"}], gold, kind


def scoring_args(row, task):
    """(k, choices) for parsing.extract: the number of options and their texts for MCQ,
    so the oracle can map stated content onto a letter without seeing the question."""
    if task == "mmlu":
        return len(row["choices"]), list(row["choices"])
    return 4, None


def render(tok, messages):
    """Chat template with the reasoning trace switched off. Templates that hard-code an
    opening <think> ignore enable_thinking, so close the block explicitly -- the same thing
    Qwen's enable_thinking=False achieves by emitting <think>\\n\\n</think>\\n\\n."""
    t = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return t + "</think>\n\n" if t.count("<think>") > t.count("</think>") else t


def budget(variant, max_new_tokens, inflation):
    fixed = VARIANTS[variant]["budget"]
    return fixed if fixed is not None else int(round(max_new_tokens * inflation))


def cell_name(model_tag, task, alpha, variant, asciiex=False, rep=1, arm=""):
    """The one place a result filename is spelled. `arm` is "" for greedy, "__smc32" or
    "__n1" for the SMC runner."""
    return (f"{model_tag}__{task}__a{alpha:g}__r{rep}{VARIANTS[variant]['suffix']}"
            f"{'__asciiex' if asciiex else ''}{arm}.jsonl")
