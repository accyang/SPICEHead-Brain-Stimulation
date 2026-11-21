# -*- coding: utf-8 -*-
"""Substitute {{marker}} placeholders in the Brain Stimulation section of the response letter.

The letter is prose, not a template — audit check 2 verifies that it quotes the live numbers. This
script is the one-time bridge: it rewrites markers in the newest section into literal values from
pipeline/values.py, so the committed letter contains numbers rather than markers.

Scoped to the section starting at SECTION so that earlier sections, which document the marker
syntax in prose, are left alone.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from values import values  # noqa: E402
from render_docs import fmt_value  # noqa: E402
from wordcount import body_words  # noqa: E402

LETTER = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "05_response_to_review.md")
TMPL = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "04_manuscript.tmpl.md")
SECTION = "# Response to the Brain Stimulation–focused review"
WORDS_BEFORE = 6453  # body word count before the BS trim, for the "down from" claim


def main():
    V = values()
    body, _ = body_words(TMPL)

    s = io.open(LETTER, encoding="utf-8").read()
    i = s.find(SECTION)
    if i < 0:
        raise SystemExit(f"{LETTER}: no '{SECTION}'")
    head, tail = s[:i], s[i:]

    extra = {"wordcount": f"{body:,}", "wordcut": f"{WORDS_BEFORE - body:,} words"}

    def sub(m):
        key, _, spec = m.group(1).partition("|")
        key = key.strip()
        if key in extra:
            return extra[key]
        if key not in V:
            raise SystemExit(f"unknown value {key!r} in the letter")
        return fmt_value(V[key], spec.strip())

    out, n = re.subn(r"\{\{([^}]*)\}\}", sub, tail)
    io.open(LETTER, "w", encoding="utf-8", newline="").write(head + out)
    print(f"filled {n} markers in the Brain Stimulation section "
          f"(body = {body} words, cut {WORDS_BEFORE - body})")


if __name__ == "__main__":
    main()
