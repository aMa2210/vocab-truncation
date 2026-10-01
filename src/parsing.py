"""Multi-tier answer extraction: strict parser first, then an LLM oracle.

Tier 1  strict   : the requested "Answer: X" format
Tier 2  oracle   : Gemini, zero-shot, T=0, acts as a pure extraction oracle -- it is
                   shown ONLY the model output and the allowed labels, never the
                   question and never the gold answer, so it cannot inject knowledge.
With the oracle switched off (use_oracle=False: run_eval --no-oracle, and run_smc.py,
which leaves the oracle to aggregate_smc.py), tier 2 is replaced by an anchored regex
layer (`lenient_no_oracle`: "answer is X", \boxed{X}, "#### X").
Every result records which tier fired.
"""
import hashlib, json, math, os, pathlib, random, re, threading, time

def _load_env():
    """Read KEY=value lines from ./api_key.env (or ./.env) so the API key never has
    to be exported by hand, and never appears in a command line or a log."""
    for d in (pathlib.Path.cwd(), pathlib.Path(__file__).resolve().parent.parent):
        for name in ("api_key.env", ".env"):
            f = d / name
            if not f.exists():
                continue
            for ln in f.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            return

_load_env()
LETTERS = "ABCDEFGH"
_client = None
ORACLE_MODEL = os.environ.get("ORACLE_MODEL", "gemini-3.5-flash")


# ---------------------------------------------------------------- deterministic
def _strict_mcq(text, k):
    # The whole output is one bare letter. Unambiguous, so it belongs in the deterministic
    # tier: the direct-answer control asks for "a single letter and nothing else", and 77%
    # of its outputs are exactly that. Sending them to the oracle would be 11,760 API calls
    # to read the string "B".
    only = re.fullmatch(r"\s*\(?([A-H])\)?[.):]?\s*", text)
    if only and LETTERS.index(only.group(1).upper()) < k:
        return only.group(1).upper()
    m = re.findall(r"(?im)^\s*answer\s*[:\-]\s*\**\s*([A-H])\b", text)
    return m[-1].upper() if m and LETTERS.index(m[-1].upper()) < k else None

# Lenient patterns are anchored on an explicit answer-announcing phrase. A bare
# "(B)" is NOT accepted: chain-of-thought math text is full of incidental
# parenthesised letters, and matching those manufactures chance-level accuracy
# out of unparseable output.
_MCQ_LENIENT = [r"(?i)\banswer\b\s*(?:is|:|-|=)?\s*\**\s*\(?([A-H])\)?\b",
                r"(?i)\bfinal\s+answer\b[^A-H\n]{0,20}\(?([A-H])\)?\b",
                r"(?i)\b(?:option|choice)\b\s*\(?([A-H])\)?\b",
                r"(?i)\\boxed\{\s*\(?([A-H])\)?\s*\}"]
# A line holding nothing but a letter, or the final number in the text, are POSITIONAL
# guesses, not answer statements. They belong after the oracle, not before it.

def _lenient_mcq(text, k, tail=1200):
    """Only the tail of the output is searched: the conclusion lives at the end."""
    valid = set(LETTERS[:k])
    seg = text[-tail:]
    for p in _MCQ_LENIENT:
        m = [x.upper() for x in re.findall(p, seg) if x.upper() in valid]
        if m: return m[-1]
    return None

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def _norm_num(s):
    s = s.replace(",", "").rstrip(".")
    # Under heavy truncation the model degenerates into very long digit runs; float()
    # turns those into inf and int(inf) raises. No benchmark answer has 16+ digits.
    if len(s.lstrip("-").split(".")[0]) > 15:
        return None
    try:
        f = float(s)
        if not math.isfinite(f):
            return None
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return None

def _strict_num(text):
    # Same for a bare number, which is what the direct-answer control asks for.
    only = re.fullmatch(r"\s*\$?(-?\d[\d,]*(?:\.\d+)?)\s*\.?\s*", text)
    if only: return _norm_num(only.group(1))
    m = re.findall(r"(?im)^\s*answer\s*[:\-]\s*\**\s*\$?(-?\d[\d,]*(?:\.\d+)?)", text)
    return _norm_num(m[-1]) if m else None

def _lenient_num(text, tail=1200):
    """Anchored patterns only -- taking the final number in the text would be a guess."""
    seg = text[-tail:]
    for p in [r"(?i)\banswer\b\s*(?:is|:|-|=)?\s*\**\s*\$?(-?\d[\d,]*(?:\.\d+)?)",
              r"(?i)\\boxed\{\s*\$?(-?\d[\d,]*(?:\.\d+)?)",
              r"####\s*(-?\d[\d,]*(?:\.\d+)?)"]:
        m = re.findall(p, seg)
        if m: return _norm_num(m[-1])
    return None


# ---------------------------------------------------------------- oracle
def _norm_ws(t):
    return re.sub(r"\s+", " ", t).strip().lower()


_COMMIT = re.compile(
    r"\b(answer|therefore|thus|hence|so|conclusion|conclude|final|correct|choose|chosen|"
    r"select|selected|option is|result is|means)\b", re.I)


def _is_option_echo(ev, letter, choices):
    """True when the quoted 'evidence' is just the question's own option line.

    Measured failure: a model that restates the choice list gets scored as committing to
    whichever option the oracle quoted -- evidence 'C. 1,2,3,4' is the prompt talking back,
    not a conclusion. The quote is genuinely in the text, so the substring check passes; it
    has to be rejected on what it *is*.
    """
    if not choices: return False
    e = _norm_ws(ev)
    if _COMMIT.search(e):                       # an actual commitment phrase: keep it
        return False
    stripped = re.sub(r"^[a-h][.):\s]+", "", e).strip()
    for c in choices:
        n = _norm_ws(c)
        if n and (stripped == n or (len(stripped) > 3 and stripped in n) or
                  (len(n) > 3 and n in stripped)):
            return True
    return False


def _oracle(text, kind, k, choices=None):
    """Ask the extraction oracle what answer the model settled on, WITH EVIDENCE.

    The oracle must quote the verbatim sentence that states the answer, and that quote is
    then checked against the text here. This is what stops it inventing an answer out of a
    degenerate output: on a repetition loop there is no sentence to quote, so it returns
    nothing. Prompt-only instructions were not enough -- the oracle would still report a
    number lifted from inside the loop.

    Two further constraints keep it from becoming a second solver:
      - the QUESTION is never shown, so it cannot work out the right answer itself;
      - for MCQ the option TEXTS are shown, so it can map "clouds begin to form" onto C.
        Verified: given the options and pure gibberish it still declines.
    """
    global _client
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not text.strip():
        return None, "oracle_unavailable"
    try:
        from google import genai
        from google.genai import types
        if _client is None:
            _client = genai.Client(api_key=key)
        body = text[-6000:]
        if kind == "mcq":
            letters = LETTERS[:len(choices)] if choices else LETTERS[:k]
            opts = ("\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
                    if choices else "(options not available)")
            prompt = (
                "Below is the raw output of a language model answering a multiple-choice "
                "question. You are shown the options but NOT the question, so you cannot and "
                "must not work out which option is correct. Report only which option the "
                "model's own text settled on -- by naming the letter, or by stating content "
                "that matches an option -- and quote the exact sentence that shows it.\n\n"
                f"OPTIONS:\n{opts}\n\n"
                'Return JSON: {"evidence": "<verbatim sentence from the text, or empty '
                'string>", "answer": "<one of ' + ", ".join(letters) + '> or null"}\n'
                "If no sentence commits to an option, return evidence \"\" and answer null. "
                "A bold marker or a stray capital letter is not evidence.\n\n"
                f"--- MODEL OUTPUT ---\n{body}\n--- END ---")
        else:
            prompt = (
                "Below is the raw output of a language model solving a problem. Report the "
                "final numeric answer it arrived at, and quote the exact sentence from the "
                "text that states it.\n\n"
                'Return JSON: {"evidence": "<verbatim sentence from the text, or empty '
                'string>", "answer": <number or null>}\n'
                "If no sentence presents a final result, return evidence \"\" and answer "
                "null. A number appearing inside a repeated phrase is not a result.\n\n"
                f"--- MODEL OUTPUT ---\n{body}\n--- END ---")

        # The API rate-limits under concurrency: at 20 workers 14 of 90 calls raised,
        # at 6 workers 5 of 90. A failed call silently becomes "unparsed", which would
        # quietly deflate accuracy, so retry with backoff instead of swallowing it.
        r = None
        for attempt in range(4):
            try:
                r = _client.models.generate_content(
                    model=ORACLE_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=2000,
                        response_mime_type="application/json"))
                break
            except Exception:
                if attempt == 3: raise
                time.sleep(2 ** attempt + random.random())
        try:
            d = json.loads(r.text or "{}")
        except Exception:
            return None, "oracle_unparsed"
        ans, ev = d.get("answer"), (d.get("evidence") or "")
        if ans is None or str(ans).strip().lower() in ("", "null", "none"):
            return None, "oracle_none"
        # the quote must really be in the output, else the oracle invented it
        probe = _norm_ws(str(ev))[:60]
        if not probe or probe not in _norm_ws(body):
            return None, "oracle_no_evidence"
        if kind == "mcq":
            m = re.search(r"[A-H]", str(ans).upper())
            if m and LETTERS.index(m.group()) < k:
                if _is_option_echo(ev, m.group(), choices):
                    return None, "oracle_option_echo"
                return m.group(), "oracle"
        else:
            m = _NUM.search(str(ans))
            # No extra gate here. Requiring the quote to carry a commitment word or an "="
            # was measured on 90 deep-alpha records: it rejected 29 of them, whose original
            # accuracy was 69% against 73% for the ones it kept -- i.e. it cut
            # indiscriminately rather than removing bad judgements, so it is not applied.
            if m: return _norm_num(m.group()), "oracle"
        return None, "oracle_unparsed"
    except Exception as e:
        return None, f"oracle_error:{type(e).__name__}"


# ---------------------------------------------------------------- entry point
def extract(text, kind, k=4, use_oracle=True, choices=None, lenient=True):
    """-> (prediction or None, tier_label). Never raises: one malformed output must not
    abort a multi-hour grid run."""
    try:
        return _extract(text, kind, k, use_oracle, choices, lenient)
    except Exception as e:
        return None, f"extract_error:{type(e).__name__}"


def _extract(text, kind, k=4, use_oracle=True, choices=None, lenient=True):
    """Two tiers: one strict deterministic parser, then the
    semantic oracle.

    The lenient regex layer that used to sit between them has been removed from scoring.
    Manual review of its extractions found it matching *deliberation* rather than
    conclusions -- "if they have different genes, that's option A. But if...", or the model
    enumerating "Option D: ..." -- because it anchors on phrases like "option X" and takes
    the last match. A pattern cannot tell a hypothesis from a conclusion; the oracle, which
    must quote a committing sentence, can. `_lenient_*` survives only as the fallback
    when the oracle is switched off.
    """
    if kind == "mcq":
        v = _strict_mcq(text, k)
        if v: return v, "strict"
    else:
        v = _strict_num(text)
        if v: return v, "strict"
    if use_oracle:
        v, tier = _oracle(text, kind, k, choices)
        if v is not None:
            return v, tier
    elif lenient:
        # only when the oracle is switched off (smoke tests): fall back to the regex layer
        v = _lenient_mcq(text, k) if kind == "mcq" else _lenient_num(text)
        if v: return v, "lenient_no_oracle"
    # If the oracle -- which is shown the whole output and only has to quote a sentence --
    # cannot find a committed answer, there is no answer to find. Guessing from position
    # was measured at 4% accuracy on repetition-loop outputs, i.e. noise, and its one
    # justification (the model computed a result but could not write the masked token
    # "Answer") is already covered by the oracle, which scores 95% on exactly those items.
    return None, "unparsed"


# ---------------------------------------------------------------- oracle cache
CACHE = "results/oracle_cache.jsonl"
_cache, _cache_lock = None, threading.Lock()


def extract_cached(text, kind, k=4, use_oracle=True, choices=None):
    """extract(), with oracle verdicts memoised on disk (CACHE, one JSON line per verdict),
    so a re-run or re-score never pays twice for the same text. The key includes the
    option texts and a version tag: change either the oracle prompt or the options and
    stale verdicts are not reused. Without the oracle, extract() is deterministic and free,
    so nothing is cached."""
    global _cache
    if not use_oracle:
        return extract(text, kind, k, use_oracle=False, choices=choices)
    key = hashlib.sha1(f"v5|{kind}|{k}|{choices}|{text}".encode()).hexdigest()
    with _cache_lock:
        if _cache is None:
            _cache = {}
            if os.path.exists(CACHE):
                with open(CACHE) as f:
                    for line in f:
                        try:
                            d = json.loads(line); _cache[d["k"]] = (d["p"], d["t"])
                        except (json.JSONDecodeError, KeyError):
                            pass                      # torn line from an interrupted run
        if key in _cache:
            return _cache[key]
    v = extract(text, kind, k, use_oracle=True, choices=choices)
    with _cache_lock:
        _cache[key] = v
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "a") as f:
            f.write(json.dumps({"k": key, "p": v[0], "t": v[1]}) + "\n")
    return v
