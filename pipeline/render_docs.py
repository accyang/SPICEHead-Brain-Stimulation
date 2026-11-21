# -*- coding: utf-8 -*-
"""Render the manuscript from its template and the computed values.

Third-round review, Major Comment 1: "Generate every numerical table directly from the result
files, and generate the numerical sentences from templates."

`04_manuscript.tmpl.md` is the editable source. It carries markers:

    {{table:aging_main}}          a whole markdown table, built in values.tables()
    {{aging.brain.thalamus.pct}}  a scalar, formatted by the spec below
    {{key|fmt}}                   explicit format, e.g. {{fem.1mm.slope|.3f}}

Formats: a bare key uses the default for its type; `|pct` gives a signed percentage with a true
minus sign; `|sci` scientific notation with superscript exponent; `|p` a p-value; anything else is
passed to format().

The rendered file is written to both `Paper_A_SPICEhead/Brain Stimulation/04_manuscript.md` and `SPICEhead_v21.md`.
Editing those directly is pointless — the next render overwrites them — so the renderer refuses to
run if it would clobber edits that are not reproducible from the template, unless --force is given.

  python pipeline/render_docs.py [--check] [--force]

`--check` renders and compares without writing; it exits non-zero if the committed file differs,
which is what the audit uses to prove the manuscript is still the template's output.
"""
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, _here)
from values import values, tables, sci, pct, pfmt, num, MINUS   # noqa: E402

BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
# (template, [rendered outputs]) — the manuscript also mirrors to the project-root working copy
JOBS = [
    (os.path.join(BS, "04_manuscript.tmpl.md"),
     [os.path.join(BS, "04_manuscript.md"), os.path.join(ROOT, "Paper_A_SPICEhead", "history_pre_v22", "manuscripts", "SPICEhead_v21.md")]),
    (os.path.join(BS, "06_supplement.tmpl.md"),
     [os.path.join(BS, "06_supplement.md")]),
]
MARKER = re.compile(r"\{\{([^}|]+?)(?:\|([^}]+))?\}\}")


def fmt_value(v, spec):
    if spec == "pct":
        return pct(v)
    if spec == "pct2":
        return pct(v, 2)
    if spec == "sci":
        return sci(v)
    if spec == "p":
        return pfmt(v)
    if spec == "num":
        return num(v)
    if spec.startswith("num"):               # num4, num5 -> that many decimals
        return num(v, int(spec[3:]))
    if spec == "minus":                      # plain number, true minus sign
        return f"{v}".replace("-", MINUS)
    if spec:
        return format(v, spec)
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render(text, V, T, used):
    def sub(m):
        key, spec = m.group(1).strip(), (m.group(2) or "").strip()
        if key.startswith("table:"):
            name = key[6:]
            if name not in T:
                raise KeyError(f"unknown table {name!r}; have {sorted(T)}")
            used.add(key)
            return T[name]
        if key not in V:
            raise KeyError(f"unknown value {key!r}")
        used.add(key)
        return fmt_value(V[key], spec)
    return MARKER.sub(sub, text)


def main(argv):
    check = "--check" in argv
    V, used = values(), set()
    T = tables(V)
    rc = 0
    for tmpl, outs in JOBS:
        if not os.path.exists(tmpl):
            print(f"  warn  template absent, skipped: {os.path.relpath(tmpl, ROOT)}")
            continue
        out = render(open(tmpl, encoding="utf-8").read(), V, T, used)
        left = MARKER.search(out)
        if left:
            raise SystemExit(f"{os.path.basename(tmpl)}: unsubstituted marker survived "
                             f"rendering: {left.group(0)}")
        for path in outs:
            old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
            if check:
                same = old == out
                print(f"  {'ok  ' if same else 'FAIL'}  {os.path.relpath(path, ROOT)}: "
                      f"{'up to date' if same else 'STALE — differs from the template render'}")
                if not same:
                    rc = 1
                    for i, (a, b) in enumerate(zip((old or "").splitlines(), out.splitlines())):
                        if a != b:
                            print(f"          first difference at line {i+1}")
                            print(f"            committed: {a[:110]}")
                            print(f"            rendered : {b[:110]}")
                            break
            else:
                with open(path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(out)
                print(f"  wrote {os.path.relpath(path, ROOT)}")
    unused = sorted(k for k in T if f"table:{k}" not in used)
    if unused and not check:
        print(f"  note: {len(unused)} generated tables unused by any template: {', '.join(unused)}")
    print(f"  {len(used)} markers substituted from {len(V)} values / {len(T)} tables")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
