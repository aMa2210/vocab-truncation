"""Vocabulary truncation via logit masking.

The index space of an lm_head splits three ways, and the split is family-specific:

  ordinary    real BPE/SP tokens                  <- alpha truncates THIS set only
  functional  structural tokens the model needs   <- always exempt
  dead        reserved/unused placeholders and
              untrained slots above the vocab     <- always blocked, at every alpha

Measured layouts:
  Qwen3.5      248,320 slots. Ordinary 0-248,043 contiguous; 33 functional at the top
               (248,044-248,076); 243 untrained slots above. eos = 248,046, i.e. 99.99%
               of the way up -- without the exemption nothing terminates below alpha 0.99.
  Llama-3.1    128,256 slots. Ordinary 0-127,999; 256 added tokens at the top of which
               only 8 are functional and 248 are <|reserved_special_token_N|>. Exempting
               all 256 would put 248 untrained tokens into the candidate set -- 43% of it
               at alpha=0.0025 -- so the placeholders must be blocked, not exempted.
  DeepSeek-R1-Distill-Llama  same layout, 13 functional / 243 placeholders.
  Gemma-4      262,144 slots. Added tokens are NEITHER contiguous NOR at the top: they are
               scattered over [0, 258,884] with <pad>=0, <eos>=1, <bos>=2 at the BOTTOM.
               The exemption is nearly a no-op here.

So the truncatable set is defined as "the ordinary tokens, ranked by index", which is
layout-independent and reduces to a simple prefix cut whenever the specials sit above the
ordinary range (Qwen, Llama).
"""
import json
import re
import torch
from transformers import LogitsProcessor

# Names that denote an unused slot rather than a token the model actually uses.
# Anchored to the full angle-bracketed form on purpose: a bare substring match on
# "unused" also hits the ordinary English word, which is a real token in every vocabulary
# here (10 of them in Qwen, 10 in Llama).
_PLACEHOLDER = re.compile(
    r"^<\|?(?:unused|reserved_special_token|extra_id|placeholder)[^>]*\|?>$", re.I)


def vocab_partition(tokenizer, model_vocab_size):
    """-> (ordinary_ids sorted, functional_ids, dead_ids)

    Placeholders are matched against the FULL vocabulary, not just the added-token table.
    Gemma-4 carries 6,227 <unusedN> entries inside its ordinary vocabulary (2.4% of the
    range, at both the bottom and the top); scanning only added_vocab leaves them in the
    truncatable set, where at low alpha they would form a large slice of the candidate pool.
    """
    full = tokenizer.get_vocab()                              # includes added tokens
    added = tokenizer.get_added_vocab()
    placeholders = {i for s, i in full.items() if _PLACEHOLDER.search(s)}
    functional = {i for s, i in added.items() if i not in placeholders}
    live = set(full.values()) - placeholders - functional
    ordinary = sorted(i for i in live if i < model_vocab_size)
    dead = sorted(set(range(model_vocab_size)) - set(ordinary) - functional)
    return ordinary, sorted(functional), dead


# Answer-format exemption. The prompt's
# format instruction demands the literal line "Answer: X", so masking any token that
# instruction requires makes the design self-contradictory at deep alpha: the model is
# ordered to write something it cannot emit. The whole channel is exempted as format
# machinery, on the same principle as the structural-token exemption.
#
# Exempting only the word is not enough: `_strict_mcq`/`_strict_num` anchor on a LINE START,
# and "on a new line" is part of the instruction, so the newline belongs to the channel too.
# Do not rely on such characters being low-index and therefore always kept: that holds for
# Qwen (keep=248 at alpha=0.001) but not for Llama, whose 128,000 ordinary ids give keep=128
# while the newline sits at id 198. The channel is therefore listed explicitly.
#
# The set stays minimal and is still content-neutral: the answer word, the layout whitespace,
# and the colon that the literal format contains. Lowercase "answer" and every other variant
# stay truncatable; letters, digits and all other punctuation stay subject to alpha (they sit
# below id 128 in both families, so they are reachable at every alpha on this grid anyway).
# All these ids stay inside the ordinary count, so keep = alpha * len(ordinary) and the
# inflation-table keys are unchanged.
_ANSWER_FORMS = ("Answer", "\u0120Answer", "\u2581Answer")   # plain, byte-BPE space, SP space

# Load-bearing: "\n" (line anchor of the strict tier) and " " (the space in "Answer: X").
# "\t"/"\r" carry no role in the required line but are layout whitespace by the same
# definition, and cost 2 ids. ":" is part of the literal format; it is below id 128 in both
# families, so exempting it changes no mask bit on this grid -- it is listed so the guarantee
# is explicit rather than incidental.
_FORMAT_CHARS = ("\n", "\t", "\r", " ", ":")


def answer_format_ids(tokenizer):
    """The 'Answer' / ' Answer' word forms only (kept for callers that want just those)."""
    full = tokenizer.get_vocab()
    return sorted(full[s] for s in _ANSWER_FORMS if s in full)


def format_char_ids(tokenizer, scan=1024):
    """Ids that decode to exactly one of _FORMAT_CHARS.

    Scans the low id range only: the single-byte/character alphabet lives there in both
    conventions (GPT-2 byte-BPE remaps the 256 bytes into ids 0-255; SentencePiece keeps
    its byte-fallback and the space marker low). Raises if the two load-bearing characters
    are not found rather than silently leaving the channel masked.
    """
    want = set(_FORMAT_CHARS)
    found = {}
    for i in range(min(scan, len(tokenizer.get_vocab()))):
        s = tokenizer.decode([i])
        if s in want and s not in found:
            found[s] = i
    for c in ("\n", " "):
        if c not in found:
            raise KeyError(f"format character {c!r} not found in the first {scan} ids; "
                           f"the truncation mask would leave the answer channel unusable")
    return sorted(found.values())


def format_channel_ids(tokenizer):
    """Everything the required answer line is made of: the word, the layout whitespace,
    the colon. This is the set exempted from truncation at every alpha."""
    return sorted(set(answer_format_ids(tokenizer)) | set(format_char_ids(tokenizer)))


# Printable ASCII plus the tab, as single-character tokens. Exempting these is a NO-OP for
# a merge-ordered BPE vocabulary -- verified, not assumed: at EVERY alpha in the grid (all
# ten, re-checked 2026-09-23) the set adds ZERO ids beyond what the alpha cut and the
# existing format channel already allow, for both Qwen3.5 and Llama-3.1. Their stored cells
# are therefore bit-identical to what this flag would produce. It is load-bearing only for
# a SentencePiece vocabulary that is not merge-ordered: Gemma-4 holds TWO copies of the
# ASCII alphabet (192 single-character ids against 96 in the BPE families) and prefers the
# copy at the 92.5th percentile when encoding, so alpha=0.9, the mildest setting in the
# grid, removes the digits, the space and the punctuation: 34.9% of the token occurrences in
# the benchmark texts, against 0.1% for Qwen at the same alpha. With the exemption the two
# families lose within ~2 points of each other at every alpha <= 0.1, in both directions
# (measured over the benchmark texts), which is what makes them comparable at all.
#
# The set is the writing system, not vocabulary: letters, digits, punctuation, space, tab.
# Control characters are excluded -- they never appear in the benchmark text and "the
# alphabet" is the easier claim to defend. Newline and carriage return are not in it either
# because Gemma already places them at ids 107 and 251; for Llama, where the newline sits at
# 198 and alpha=0.001 keeps only 128 tokens, it is the format channel above that exempts
# them, and that mechanism is unchanged.
#
# CAVEAT: this is an English alphabet. All three benchmarks are English, so it
# covers their writing system exactly; a Chinese benchmark would need the same argument made
# for a different character set, and "which characters are primitive" stops being obvious.
def ascii_primitive_ids(tokenizer):
    """Ids that decode to exactly one printable-ASCII character, or to a tab."""
    full = tokenizer.get_vocab()
    out = []
    for s, i in full.items():
        d = tokenizer.convert_tokens_to_string([s])
        if len(d) == 1 and d.isascii() and (d.isprintable() or d == "\t"):
            out.append(i)
    return sorted(out)


def exempt_ids(tokenizer, exempt_ascii=False):
    """Ids allowed at every alpha on top of the functional tokens: the answer-format
    channel, plus the single-character ASCII alphabet when `exempt_ascii` (gemma-4)."""
    ids = set(format_channel_ids(tokenizer))
    if exempt_ascii:
        ids |= set(ascii_primitive_ids(tokenizer))
    return sorted(ids)


def allowed_ids(ordinary, functional, alpha, extra=()):
    """The allow-set for one alpha: the lowest-ranked alpha share of the ordinary tokens,
    plus the functional and exempt ids. Returns (sorted ids, n_ordinary_kept). Both the
    greedy logits mask and the SMC potential are built from this, so the two decoders
    always truncate to the same set."""
    keep = int(alpha * len(ordinary))
    return sorted(set(ordinary[:keep]) | set(functional) | set(extra)), keep


def build_block_mask(ordinary, functional, model_vocab_size, alpha, device, extra_allow=()):
    """Boolean mask, True = clamp this logit to -inf. Returns (mask, n_kept). Dead ids
    (placeholders, untrained slots) are never in the allow-set, so they stay blocked."""
    ids, keep = allowed_ids(ordinary, functional, alpha, extra_allow)
    allow = torch.zeros(model_vocab_size, dtype=torch.bool, device=device)
    allow[torch.tensor(ids, device=device)] = True
    return ~allow, keep


class VocabTruncationProcessor(LogitsProcessor):
    """Clamp masked logits before softmax. Generation steps only; prompts are untouched."""

    def __init__(self, block_mask):
        self.block = block_mask

    def __call__(self, input_ids, scores):
        return scores.masked_fill(self.block, torch.finfo(scores.dtype).min)


def describe(alpha, keep, n_ordinary, n_functional, n_dead):
    return (f"alpha={alpha:<7} kept={keep}/{n_ordinary} ordinary "
            f"({100*keep/n_ordinary:.2f}%)  +{n_functional} functional exempt  "
            f"-{n_dead} dead blocked")


_INFL = None
def inflation(task, alpha, n_ordinary):
    """Token-budget multiplier for (task, alpha) from analysis/inflation.json, keyed by the
    tokenizer's ordinary-token count. A truncated vocabulary needs more tokens to write the
    same text, so the budget grows with it and "ran out of room" is never scored as "could
    not express it". The table is per tokenizer: one family's factors applied to another
    would reintroduce the confound, so a missing tokenizer falls back to 1.0 with a warning
    rather than borrowing numbers."""
    global _INFL
    if _INFL is None:
        with open("analysis/inflation.json") as f:
            _INFL = json.load(f)
    fam = _INFL.get(str(n_ordinary))
    if fam is None:
        print(f"[warn] no inflation table for tokenizer with {n_ordinary} ordinary tokens; "
              f"budget scaling disabled", flush=True)
        return 1.0
    val = fam.get(task, {}).get(f"{alpha:g}")
    if val is None:
        raise KeyError(f"no inflation factor for task={task} alpha={alpha:g} (tokenizer key "
                       f"{n_ordinary}); add it with analysis/make_inflation.py, or a fixed "
                       f"budget silently under-budgets this alpha")
    return float(val)
