"""Greedy arm: vocabulary-truncated generation under each instruction variant.

python src/run_eval.py --model Qwen/Qwen3.5-9B --tasks gsm8k mmlu \
       --alphas 1 0.1 0.05 0.001 --prompts cot simple nocot

One process loads the model once and runs every (prompt, task, alpha) cell, writing one
JSONL per cell (results/<model>__<task>__a<alpha>__r<rep><suffix>.jsonl, see
prompts.cell_name). Existing cells are skipped.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

# Greedy decoding is NOT bit-reproducible on this stack (measured: ~80/100 items differ
# between identical runs). These knobs do not eliminate it, but they cut the run-to-run
# accuracy spread from ~5.0 to ~2.0 points, and cost nothing. Must be set before CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# A grid is a sequence of allocations whose size changes every cell, which is the case
# expandable segments exist for.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as PR, parsing as PA
from truncation import (vocab_partition, build_block_mask, VocabTruncationProcessor,
                        describe, exempt_ids, inflation)


def rep_ratio(ids, n=8):
    """Share of repeated 8-grams: a looping generation scores near 1."""
    if len(ids) < 2 * n: return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", choices=list(PR.TASK_FILES), default=["gsm8k", "mmlu"])
    ap.add_argument("--alphas", nargs="+", type=float, default=list(PR.ALPHAS))
    ap.add_argument("--prompts", nargs="+", choices=list(PR.VARIANTS), default=["cot"],
                    help="instruction variants, see prompts.VARIANTS")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 100 items")
    ap.add_argument("--budget-mode", choices=["fixed", "scaled"], default="scaled",
                    help="scaled: multiply max-new-tokens by the per-(task,alpha) "
                         "sequence-inflation factor, so every alpha gets an equal budget "
                         "in reference-text-equivalent terms")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rep", type=int, default=1,
                    help="repeat index; generation is not bit-reproducible on this stack, "
                         "so repeats quantify the noise floor")
    ap.add_argument("--exempt-ascii", action="store_true",
                    help="Additionally exempt every single-character printable-ASCII token "
                         "(and the tab) from truncation. A no-op for Qwen3.5, whose alphabet "
                         "already sits below id 256; needed for gemma-4, which keeps the "
                         "alphabet it actually encodes with at the 92.5th percentile, so that "
                         "alpha=0.9 removes the digits, the space and the punctuation -- "
                         "34.9%% of benchmark token occurrences against 0.1%% for Qwen.")
    a = ap.parse_args()

    tag = a.model.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    path = lambda p, t, al: os.path.join(a.out, PR.cell_name(tag, t, al, p, a.exempt_ascii, a.rep))
    planned = [path(p, t, al) for p in a.prompts for t in a.tasks for al in a.alphas]
    if not a.overwrite and all(os.path.exists(p) for p in planned):
        print(f"[done] all {len(planned)} cells already present, not loading the model",
              flush=True)
        return

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print(f"[load] {a.model}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    V = model.get_output_embeddings().weight.shape[0]
    print(f"[load] {time.time()-t0:.0f}s  lm_head vocab={V}", flush=True)

    ordinary, functional, dead = vocab_partition(tok, V)
    extra = exempt_ids(tok, a.exempt_ascii)
    print(f"[vocab] ordinary={len(ordinary)} [{ordinary[0]},{ordinary[-1]}]  "
          f"functional={len(functional)}  dead={len(dead)}  always-allowed={len(extra)}",
          flush=True)

    # eos set: the chat end marker, plus <|endoftext|> when the vocabulary has one
    eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")} - {None})

    for variant in a.prompts:
        for task in a.tasks:
            rows = PR.load_rows(task, a.limit)
            built = [PR.build(r, task, variant) for r in rows]
            sargs = [PR.scoring_args(r, task) for r in rows]
            texts = [PR.render(tok, b[0]) for b in built]

            for alpha in a.alphas:
                fn = path(variant, task, alpha)
                if os.path.exists(fn) and not a.overwrite:
                    print(f"[skip] {fn}", flush=True); continue

                infl = inflation(task, alpha, len(ordinary)) if a.budget_mode == "scaled" else 1.0
                mnt = PR.budget(variant, a.max_new_tokens, infl)
                block, keep = build_block_mask(ordinary, functional, V, alpha, model.device,
                                               extra_allow=extra)
                blocked = set(block.nonzero().flatten().tolist())  # one device copy per cell
                proc = LogitsProcessorList([VocabTruncationProcessor(block)])
                print(f"[run] {tag} {variant} {task} " +
                      describe(alpha, keep, len(ordinary), len(functional), len(dead)) +
                      f"  budget={mnt} (x{infl:.2f})", flush=True)

                t1, gen = time.time(), []
                for i in range(0, len(texts), a.batch_size):
                    # apply_chat_template already emitted the template's own BOS, so no
                    # special tokens are added here (a second BOS breaks Llama-style models).
                    enc = tok(texts[i:i + a.batch_size], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(model.device)
                    with torch.inference_mode():
                        out = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                             logits_processor=proc,
                                             pad_token_id=tok.pad_token_id, eos_token_id=eos)
                    for raw in out[:, enc["input_ids"].shape[1]:].tolist():
                        # Trim only TRAILING padding: filtering every pad would also delete
                        # the model's own stop token whenever pad == eos. Having trimmed
                        # anything is itself the signal that generation stopped early.
                        end = len(raw)
                        while end and raw[end - 1] == tok.pad_token_id:
                            end -= 1
                        ids = raw[:end]
                        bad = [t for t in ids if t in blocked]
                        assert not bad, (f"MASK LEAK at alpha={alpha}: emitted blocked ids "
                                         f"{bad[:5]} ({[tok.decode([b]) for b in bad[:5]]})")
                        stopped = end < len(raw) or (bool(ids) and ids[-1] in eos)
                        gen.append((ids, tok.decode(ids, skip_special_tokens=True), stopped))
                    print(f"    {len(gen)}/{len(texts)}  {time.time()-t1:.0f}s", flush=True)

                # The oracle is network-bound, so extract in a thread pool or low-alpha
                # cells spend minutes serialised on the API.
                with ThreadPoolExecutor(max_workers=8) as ex:
                    extracted = list(ex.map(
                        lambda i: PA.extract_cached(gen[i][1], built[i][2], sargs[i][0],
                                                    use_oracle=not a.no_oracle,
                                                    choices=sargs[i][1]),
                        range(len(gen))))

                recs = []
                for idx, ((ids, txt, stopped), (pred, tier)) in enumerate(zip(gen, extracted)):
                    gold = built[idx][1]
                    recs.append(dict(idx=idx, alpha=alpha, task=task, model=tag, rep=a.rep,
                                     cond=variant, think="off", style="chat",
                                     asciiex=int(a.exempt_ascii), quant="none", gold=gold,
                                     pred=pred, tier=tier,
                                     correct=int(pred is not None and pred == gold),
                                     n_gen=len(ids), hit_max=int(not stopped and len(ids) >= mnt),
                                     budget=mnt, eos=int(stopped),
                                     rep8=round(rep_ratio(ids), 4), forced=0, output=txt))
                with open(fn, "w") as f:
                    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n = len(recs)
                print(f"[done] {fn}  acc={sum(r['correct'] for r in recs)/n:.3f}  "
                      f"unparsed={sum(r['pred'] is None for r in recs)/n:.2f}  "
                      f"hit_max={sum(r['hit_max'] for r in recs)/n:.2f}  "
                      f"{time.time()-t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
