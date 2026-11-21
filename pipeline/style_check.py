# -*- coding: utf-8 -*-
"""Check the manuscript and supplement against unified-academic-style-prompt.md.

The rules that can be checked mechanically:
  * no dash used as sentence punctuation (em dash, or spaced en dash joining clauses);
    numeric ranges with an unspaced en dash are fine, as are hyphenated compounds
  * the empirical avoid list
  * the sparing list ("very", "novel", "clearly", "interestingly")
  * sentence length: median in the low 20s, nothing much past 45 words

  python pipeline/style_check.py [file ...]
"""
import io
import os
import re
import statistics as st
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
DEFAULT = [os.path.join(BS, "04_manuscript.md"), os.path.join(BS, "06_supplement.md")]

AVOID = [
    (r"due to the fact that", "because"),
    (r"\bobviously\b", "delete, or 'as expected'"),
    (r"\bfirstly\b|\bsecondly\b|\bthirdly\b", "First, Second, Third"),
    (r"it should be noted that|it is worth noting that", "state the point directly"),
    (r"we believe|we feel|we think\b", "'these data suggest', or state it plainly"),
    (r"\bprove[ds]?\b|\bproof\b", "demonstrate, show, establish"),
    (r"\bdramatically\b", "give the magnitude"),
    (r"\bremarkably\b|\bstrikingly\b", "delete, or give the number"),
    (r"\bparadigm\b|groundbreaking|state of the art|cutting edge", "describe what is new"),
    (r"\bquite\b|\brather\b(?! than)", "delete the intensifier"),
    (r"in order to", "to"),
]
SPARING = [r"\bvery\b", r"\bnovel\b", r"\butilize[sd]?\b", r"\bclearly\b", r"\binterestingly\b"]


def prose(text):
    """Body prose only: drop tables, headings, code fences, blockquote callouts.

    The reference list is excluded too: those are published titles, and a dash inside someone
    else's title is theirs to keep.
    """
    # Stop at the end matter. Acknowledgements onward is either wording Elsevier mandates
    # verbatim (the generative-AI declaration template contains "in order to") or ⚠️ instructions
    # to the authors, and the style prompt defers to the journal where the two conflict.
    for cut in ("\n## Acknowledgements", "\n## References"):
        text = text.split(cut, 1)[0]
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    # "Writing – original draft" and "Writing – review and editing" are the official CRediT role
    # names; the en dash there is part of the taxonomy label, not sentence punctuation.
    text = re.sub(r"Writing\s+–\s+(original draft|review and editing)", r"Writing \1", text)
    out = []
    for line in text.split("\n"):
        q = line.strip()
        if not q or q.startswith("#") or q.startswith("|") or q.startswith(">"):
            continue
        if set(q) <= set("-|: "):
            continue
        out.append(q)
    return " ".join(out)


def sentences(p):
    p = re.sub(r"\b([A-Z]|Fig|et al|e\.g|i\.e|vs|approx|ca|Dr|Eq)\.", r"\1<DOT>", p)
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", p) if x.strip()]
    return [x.replace("<DOT>", ".") for x in parts]


def check(path):
    raw = io.open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    p = prose(raw)
    print(f"\n=== {name} ===")

    # --- dash as sentence punctuation ---
    # Only prose counts: the rule is about dashes linking clauses or setting off asides. A lone
    # "—" standing for a missing value in a table cell is not sentence punctuation.
    em = re.findall(r"[^\n]{0,45}—[^\n]{0,45}", p)
    sp_en = re.findall(r"[^\n]{0,45}\s–\s[^\n]{0,45}", p)
    in_tables = len(re.findall("—", raw)) - len(re.findall("—", p))
    print(f"  em dash in prose (—)   : {len(em)}   (+{in_tables} in tables/headings, allowed)")
    print(f"  spaced en dash ( – )   : {len(sp_en)}")
    for s in em[:4]:
        print(f"      … {s.strip()} …")

    # --- avoid list ---
    hits = [(pat, sub, len(re.findall(pat, p, re.I))) for pat, sub in AVOID]
    bad = [(a, b, c) for a, b, c in hits if c]
    print(f"  avoid-list hits        : {sum(c for _, _, c in bad)}")
    for pat, sub, c in bad:
        found = set(m.lower() for m in re.findall(pat, p, re.I) if isinstance(m, str) and m)
        print(f"      {c}× /{pat}/ {sorted(found) if found else ''} -> {sub}")

    # --- sparing list ---
    for pat in SPARING:
        c = len(re.findall(pat, p, re.I))
        if c:
            print(f"  sparing: {pat:24s} {c}×")

    # --- sentence length ---
    ss = [len(x.split()) for x in sentences(p) if len(x.split()) > 2]
    if ss:
        long_ = sorted((n for n in ss if n > 45), reverse=True)
        print(f"  sentences {len(ss)}  median {st.median(ss):.0f}  mean {st.mean(ss):.0f}  "
              f"max {max(ss)}")
        print(f"  over 45 words          : {len(long_)} {long_[:8]}")
    return len(em) + len(sp_en)


if __name__ == "__main__":
    paths = sys.argv[1:] or DEFAULT
    total = sum(check(p) for p in paths)
    print(f"\ndashes used as sentence punctuation, all files: {total}")
    raise SystemExit(1 if total else 0)
