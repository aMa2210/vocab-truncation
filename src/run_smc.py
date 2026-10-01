"""SMC decoding arm of the prompt experiment.

Same construct as run_eval.py -- the same allow-set (truncation.allowed_ids), prompts,
task files and token budgets -- but the decoder is genlm-control's Sequential Monte Carlo
(AWRS proposal + resampling when ESS < N/2) instead of greedy local masking.
n_particles=1 is a single masked sample (the "n1" arm).

Runs in its own environment (genlm-control 0.4.0 + genlm-backend 0.3.0 + vllm 0.26 +
transformers 5.17; see README): `venv_smc/bin/python src/run_smc.py ...`.
smc_compat.apply_v2() is mandatory: the released genlm-backend 0.3.0 crashes on its first
forward pass without it.

Scoring here is the strict tier only (no oracle, no regex fallback), the same first tier
run_eval.py applies. Every particle's raw text is stored, and aggregate_smc.py
re-scores with the oracle in the main environment, so the Gemini dependency stays out of
this one and each unique text is scored (and billed) once.

Determinism: seed = md5(model|task|alpha|prompt|rep|n_particles), and question i uses
seed+i for both RNGs (np.random for SMC resampling, AWRS for token draws).
"""
import argparse, asyncio, hashlib, json, os, sys, time
from importlib.metadata import version

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import smc_compat
smc_compat.apply_v2()

import numpy as np

import prompts as PR
import parsing as PA
from truncation import vocab_partition, exempt_ids, allowed_ids, inflation

# hit_max below relies on 0.4.0's forced EOS at max_tokens; under 0.3.x it would be wrong.
_GENLM, _VLLM = version("genlm-control"), version("vllm")
assert tuple(map(int, _GENLM.split(".")[:2])) >= (0, 4), f"needs genlm-control>=0.4, got {_GENLM}"
ENGINE = f"genlm{'.'.join(_GENLM.split('.')[:2])}-vllm{'.'.join(_VLLM.split('.')[:2])}"


def cell_seed(model, task, alpha, prompt, rep, n_particles):
    key = f"{model}|{task}|{alpha:g}|{prompt}|{rep}|{n_particles}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def alpha_mask(Potential, vocab, allowed):
    """Boolean genlm potential: log-weight 0 for allowed token ids (and EOS), -inf else."""
    class _Mask(Potential):
        def __init__(self):
            super().__init__(vocab)
            self.allowed = frozenset(allowed)
            logws = self.alloc_logws()
            for pos, t in enumerate(self.vocab):
                if t.token_id in self.allowed:
                    logws[pos] = 0.0
            logws[-1] = 0.0                           # EOS, functional-exempt
            self._logws = self.make_lazy_weights(logws)

        async def prefix(self, context):
            # Every earlier token was accepted under this same potential when it was
            # appended, so only the newest one can change the verdict. Re-checking the
            # whole context made each call O(length).
            ok = not context or context[-1].token_id in self.allowed
            return 0.0 if ok else float("-inf")

        async def complete(self, context):
            return await self.prefix(context)

        async def logw_next(self, context):
            return self._logws
    return _Mask()


async def run(a):
    from genlm.control import PromptedLLM, AWRS, Potential
    from transformers import AutoTokenizer, AutoConfig

    tag = a.model.split("/")[-1]
    os.makedirs(a.out, exist_ok=True)
    arm = lambda n: f"__smc{n}" if n > 1 else "__n1"
    path = lambda p, t, al, n: os.path.join(
        a.out, PR.cell_name(tag, t, al, p, a.exempt_ascii, a.rep, arm(n)))
    todo = [path(p, t, al, n) for p in a.prompts for t in a.tasks for al in a.alphas
            for n in a.particles]
    if not a.overwrite and all(os.path.exists(f) for f in todo):
        print(f"[skip-all] {tag} prompts={a.prompts}: {len(todo)} cells present, "
              f"no model load", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = AutoConfig.from_pretrained(a.model)
    V = getattr(cfg, "vocab_size", None) or cfg.get_text_config().vocab_size
    ordinary, functional, _ = vocab_partition(tok, V)
    extra = exempt_ids(tok, a.exempt_ascii)

    # eos set mirrors run_eval: chat end marker + <|endoftext|>
    eos_strings = sorted({tok.decode([tok.eos_token_id]), "<|endoftext|>"}
                         & set(tok.get_vocab()))
    llm = PromptedLLM.from_name(
        a.model,
        eos_byte_strings=[s.encode() for s in eos_strings],
        engine_opts={"gpu_memory_utilization": a.gpu_mem, "max_model_len": 6144},
    )
    print(f"[load] {a.model}  vocab={len(llm.vocab)}  eos={eos_strings}  {ENGINE}", flush=True)

    for variant in a.prompts:
        for task in a.tasks:
            rows = PR.load_rows(task, a.limit)
            built = [PR.build(r, task, variant) for r in rows]
            sargs = [PR.scoring_args(r, task) for r in rows]
            prompt_ids = [tok(PR.render(tok, b[0]), add_special_tokens=False)["input_ids"]
                          for b in built]

            for alpha in a.alphas:
                allowed, keep = allowed_ids(ordinary, functional, alpha, extra)
                mnt = PR.budget(variant, a.max_new_tokens, inflation(task, alpha, len(ordinary)))
                mask = None                               # built once per alpha, on demand
                for n_p in a.particles:
                    fn = path(variant, task, alpha, n_p)
                    if os.path.exists(fn) and not a.overwrite:
                        print(f"[skip] {fn}", flush=True)
                        continue
                    if mask is None:
                        mask = alpha_mask(Potential, llm.vocab, allowed)
                    seed0 = cell_seed(a.model, task, alpha, variant, a.rep, n_p)
                    print(f"[run] {fn}  keep={keep}/{len(ordinary)} budget={mnt} "
                          f"n={n_p} seed={seed0}", flush=True)
                    await run_cell(a, AWRS, llm, mask, set(allowed), fn, rows, built, sargs,
                                   prompt_ids, tag, task, alpha, variant, n_p, mnt, seed0)


async def run_cell(a, AWRS, llm, mask, allowed, fn, rows, built, sargs, prompt_ids,
                   tag, task, alpha, variant, n_p, mnt, seed0):
    # Per-question checkpoint: questions append to <fn>.partial as they finish, and a
    # relaunch resumes at the first missing one. Because question i always runs on
    # seed0+i, a resumed cell is byte-identical to an uninterrupted one. The .partial is
    # renamed to <fn> when complete.
    part_fn = fn + ".partial"
    recs = []
    if os.path.exists(part_fn):
        with open(part_fn) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    break                              # torn last line from the kill
        recs = [r for j, r in enumerate(recs) if r["idx"] == j]
        with open(part_fn, "w") as f:                  # drop any torn tail
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"    resume at item {len(recs)}/{len(rows)}", flush=True)
    t0 = time.time()
    for i in range(len(recs), len(rows)):
        _, gold, kind = built[i]
        k, _choices = sargs[i]
        llm.prompt_ids = prompt_ids[i]
        np.random.seed((seed0 + i) % 2**32)
        sampler = AWRS(llm, mask, seed=(seed0 + i) % 2**32)
        seqs = await sampler.smc(n_particles=n_p, ess_threshold=0.5, max_tokens=mnt)

        parts, votes = [], {}
        for ctx, logw, w in zip(seqs.contexts, seqs.log_weights, seqs.normalized_weights):
            ids = [t.token_id for t in ctx if hasattr(t, "token_id")]
            # A particle AWRS killed carries logw = -inf and ends in the rejected token it
            # was handed back (nothing accepted, or the allowed set holds < 1e-5 of the
            # mass). That token is not a leak -- the particle has zero weight. Only a live
            # particle with a banned id is one.
            dead = not np.isfinite(float(logw))
            bad = [t for t in ids if t not in allowed]
            assert dead or not bad, f"MASK LEAK {fn} item {i}: {bad[:5]}"
            if dead and bad:
                assert ids and ids[-1] in bad and len(bad) == 1, \
                    f"dead particle with banned ids before its last token {fn} item {i}"
            txt = b"".join(t.byte_string for t in ctx
                           if hasattr(t, "byte_string")).decode("utf-8", "replace")
            # genlm-control 0.4.0 forces EOS when one token of budget is left, so a
            # particle cut off by the budget has exactly mnt-1 ids and a natural stop at
            # most mnt-2: the test below is exact, not a margin. Counting a cut-off
            # particle as a natural stop would report looping particles as terminated.
            hit_max = len(ids) >= mnt - 1
            pred, tier = PA.extract(txt, kind, k, use_oracle=False, lenient=False)
            w = 0.0 if dead or not np.isfinite(float(w)) else float(w)
            if pred is not None and not dead:
                votes[pred] = votes.get(pred, 0.0) + w
            parts.append(dict(output=txt, logw=float(logw), w=w,
                              n_gen=len(ids), hit_max=int(hit_max),
                              eos=int(not hit_max and not dead),
                              dead=int(dead), pred=pred, tier=tier))

        pred_vote = max(votes, key=votes.get) if votes else None
        live = [p for p in parts if not p["dead"]]
        recs.append(dict(
            idx=i, alpha=alpha, task=task, model=tag, rep=a.rep, cond=variant,
            style="chat", asciiex=int(a.exempt_ascii), engine=ENGINE, n_particles=n_p,
            ess_threshold=0.5, seed=(seed0 + i) % 2**32, budget=mnt,
            gold=gold, kind=kind, pred=pred_vote,
            correct=int(pred_vote is not None and pred_vote == gold),
            post_correct_mass=round(float(sum(p["w"] for p in live if p["pred"] == gold)), 4),
            eos=round(sum(p["eos"] * p["w"] for p in live), 4),
            n_dead=len(parts) - len(live),
            diversity=len({p["output"] for p in parts}),
            particles=parts))
        with open(part_fn, "a") as f:
            f.write(json.dumps(recs[-1]) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    assert len(recs) == len(rows) and all(r["idx"] == j for j, r in enumerate(recs))
    os.replace(part_fn, fn)
    print(f"[done] {fn}  acc(vote,pre-oracle)={sum(r['correct'] for r in recs)/len(recs):.3f}"
          f"  {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="+", choices=list(PR.TASK_FILES), default=["gsm8k", "mmlu"])
    ap.add_argument("--alphas", nargs="+", type=float, default=list(PR.ALPHAS))
    ap.add_argument("--particles", nargs="+", type=int, default=[32, 1])
    ap.add_argument("--prompts", nargs="+", choices=list(PR.VARIANTS), default=["cot"])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results_smc")
    ap.add_argument("--exempt-ascii", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--overwrite", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
