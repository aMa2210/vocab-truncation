"""Oracle re-scoring and summary for the SMC decoder arm.

run_smc.py scores particles without the oracle, so the Gemini dependency stays out of the
SMC environment. The greedy numbers (run_eval.py) are oracle-scored, and the oracle moves
greedy accuracy by up to +0.11 at alpha=0.1-0.05 -- the same size as the SMC-vs-sampling
gaps -- so the SMC arm must be scored the same way.

--rescore (needs the full results_smc/ from run_smc_grid.sh): every LIVE particle the
strict tier did not parse goes through parsing.extract_cached exactly as run_eval.py calls
it (same oracle, same cache, MMLU option texts passed) -- including any particle a regex
fallback parsed, since greedy outputs never pass through that layer. Strict-parsed
particles keep their answer.
Dead particles (AWRS-killed, logw=-inf) stay out of the vote. Each cell's vote,
correctness and posterior-correct mass are recomputed and written to
results_smc_oracle/<same name>, one cell at a time; existing outputs are skipped
(--overwrite re-scores them). strict_pred / strict_correct keep the strict tier alone,
the same thing greedy's strict column measures.

Default: the greedy / n1 / smc32 table per (model, prompt, task, alpha), in both scorings,
with each decoder's LFC against its own alpha=1 -> results_smc_oracle/summary.json
"""
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as PR

SRC, OUT = "results_smc", "results_smc_oracle"
ARMS = ("greedy", "n1", "smc32")


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def rescore_cell(src, dst, workers):
    import parsing as PA
    recs = read(src)
    rows = PR.load_rows(recs[0]["task"])
    jobs = {}
    for r in recs:
        k, ch = PR.scoring_args(rows[r["idx"]], r["task"])
        for p in r["particles"]:
            if not p.get("dead") and p["tier"] != "strict":
                jobs[(p["output"], r["kind"], k, repr(ch))] = (p["output"], r["kind"], k, ch)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = dict(zip(jobs, ex.map(
            lambda j: PA.extract_cached(j[0], j[1], j[2], use_oracle=True, choices=j[3]),
            jobs.values())))

    for r in recs:
        k, ch = PR.scoring_args(rows[r["idx"]], r["task"])
        votes, strict_votes = {}, {}
        for p in r["particles"]:
            p["strict_tier"] = p["tier"]
            p["strict_pred"] = p["pred"] if p["tier"] == "strict" else None
            if p.get("dead"):
                continue
            if p["tier"] != "strict":
                p["pred"], p["tier"] = verdicts[(p["output"], r["kind"], k, repr(ch))]
            if p["pred"] is not None:
                votes[p["pred"]] = votes.get(p["pred"], 0.0) + p["w"]
            if p["strict_pred"] is not None:
                strict_votes[p["strict_pred"]] = strict_votes.get(p["strict_pred"], 0.0) + p["w"]
        r["strict_pred"] = max(strict_votes, key=strict_votes.get) if strict_votes else None
        r["strict_correct"] = int(r["strict_pred"] is not None and r["strict_pred"] == r["gold"])
        r["pred"] = max(votes, key=votes.get) if votes else None
        r["correct"] = int(r["pred"] is not None and r["pred"] == r["gold"])
        r["post_correct_mass"] = round(sum(p["w"] for p in r["particles"]
                                           if not p.get("dead") and p["pred"] == r["gold"]), 4)
    with open(dst, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return len(jobs)


def rescore(workers, overwrite=False):
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{SRC}/*.jsonl"))
    todo = [f for f in files
            if overwrite or not os.path.exists(f"{OUT}/{os.path.basename(f)}")]
    print(f"{len(files)} cells, {len(todo)} to rescore", flush=True)
    for n, f in enumerate(todo, 1):
        calls = rescore_cell(f, f"{OUT}/{os.path.basename(f)}", workers)
        print(f"    [{n}/{len(todo)}] {os.path.basename(f)}: {calls} unparsed texts", flush=True)
    print(f"[rescored] {len(todo)} cells", flush=True)


def cell_stats(path, strict_of):
    rs = read(path)
    n = len(rs)
    return (sum(r["correct"] for r in rs) / n, sum(strict_of(r) for r in rs) / n,
            sum(r["eos"] for r in rs) / n)


def summary():
    greedy_strict = lambda r: int(r["correct"] and r["tier"].startswith("strict"))
    smc_strict = lambda r: r["strict_correct"]
    rows = []
    for repo, _, asciiex, has_smc in PR.MODELS:
        if not has_smc:
            continue
        tag = repo.split("/")[-1]
        for variant in PR.VARIANTS:
            for task in PR.TASK_FILES:
                for alpha in PR.ALPHAS:
                    cell = lambda arm: PR.cell_name(tag, task, alpha, variant, asciiex, arm=arm)
                    cells = {"greedy": (f"results/{cell('')}", greedy_strict),
                             "n1":     (f"{OUT}/{cell('__n1')}", smc_strict),
                             "smc32":  (f"{OUT}/{cell('__smc32')}", smc_strict)}
                    if not all(os.path.exists(p) for p, _ in cells.values()):
                        continue
                    row = dict(model=tag, prompt=variant, task=task, alpha=float(alpha))
                    for arm, (p, strict_of) in cells.items():
                        row[arm], row[arm + "_strict"], row[arm + "_eos"] = cell_stats(p, strict_of)
                    rows.append(row)

    # LFC against each decoder's own alpha=1, in both scorings
    base = {(r["model"], r["prompt"], r["task"]): r for r in rows if r["alpha"] == 1.0}
    for r in rows:
        b = base.get((r["model"], r["prompt"], r["task"]))
        if not b:
            continue
        for k in (arm + s for arm in ARMS for s in ("", "_strict")):
            r["lfc_" + k] = round(r[k] / b[k], 3) if b[k] else None
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(rows, f, indent=1)

    group = None
    for r in rows:
        if (r["model"], r["task"], r["prompt"]) != group:
            group = (r["model"], r["task"], r["prompt"])
            print(f"== {r['model']} {r['task']} prompt={r['prompt']}   "
                  f"oracle acc (LFC) | strict acc | eos")
        print(f"  a={r['alpha']:<6g}" + "".join(
            f"  {arm:6s} {r[arm]:.2f} ({r.get('lfc_' + arm) or 0:.2f}) "
            f"s{r[arm + '_strict']:.2f} e{r[arm + '_eos']:.2f}" for arm in ARMS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="re-score cells already scored")
    a = ap.parse_args()
    if a.rescore:
        rescore(a.workers, a.overwrite)
    summary()
