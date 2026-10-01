"""Per-(tokenizer, task, alpha) sequence-length inflation -> analysis/inflation.json.

A truncated vocabulary needs more tokens to write the same text, so the generation budget
must grow with it, or "ran out of room" would be scored as "cannot express the concept".
For each alpha this greedily re-tokenises reference text (the first 50 questions of each
task) using only the allowed tokens and records (tokens needed) / (tokens normally used).

Keyed by the number of ordinary tokens, which identifies a tokenizer: Qwen3.5-2B and -9B
share key 248044, every gemma-4 size shares 255893. Run from the bundle root:
    venv/bin/python analysis/make_inflation.py [repo ...]
Passing repo names recomputes only those tokenizers and leaves the other keys untouched.
"""
import json, os, sys
sys.path.insert(0, "src")
from transformers import AutoTokenizer
import prompts as PR
from truncation import vocab_partition

FAM = {"Qwen/Qwen3.5-9B": 248320,           # repo -> lm_head size (incl. unused slots)
       "google/gemma-4-E4B-it": 262144}
if len(sys.argv) > 1:
    FAM = {k: v for k, v in FAM.items() if k in sys.argv[1:]}
    assert FAM, f"none of {sys.argv[1:]} is a known tokenizer; known: {list(FAM)}"
ALPHAS = [1.0, .9, .8, .7, .6, .5, .4, .35, .3, .25, .2, .15, .1, .075, .05, .025, .01,
          .005, .0025, .001]

m, g = PR.load_rows("mmlu"), PR.load_rows("gsm8k")
SRC = {"mmlu":  [r["question"] + " " + " ".join(r["choices"]) for r in m][:50],
       "gsm8k": [r["answer"] for r in g][:50]}

path = "analysis/inflation.json"
out = json.load(open(path)) if os.path.exists(path) else {}

for repo, V in FAM.items():
    tok = AutoTokenizer.from_pretrained(repo)
    ordinary, _, _ = vocab_partition(tok, V)
    key = str(len(ordinary))
    order = [(i, tok.decode([i])) for i in ordinary]
    tab = {t: {} for t in SRC}
    print(f"=== {repo}  ordinary={len(ordinary)} key={key}", flush=True)
    for al in ALPHAS:
        keep = max(int(al * len(ordinary)), 1)
        sub = {s: i for i, s in order[:keep] if s}
        maxlen = max((len(s) for s in sub), default=1)
        for t, texts in SRC.items():
            base = sum(len(tok(x, add_special_tokens=False).input_ids) for x in texts)
            n = 0
            for x in texts:
                p = 0
                while p < len(x):
                    for L in range(min(maxlen, len(x) - p), 0, -1):
                        if x[p:p+L] in sub: p += L; n += 1; break
                    else: p += 1; n += 1
            tab[t][f"{al:g}"] = round(n / base, 4)
        print(f"  alpha={al:<7} " + "  ".join(f"{t}:{tab[t][f'{al:g}']:.2f}x" for t in SRC), flush=True)
    out[key] = tab
json.dump(out, open(path, "w"), indent=1)
print(f"\nwrote {path}  keys:", list(out))
