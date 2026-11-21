# -*- coding: utf-8 -*-
"""Render the v23 Results record from its template and values_v23.

Paper A's final results are v23 (PI, 26 October 2026). The manuscript is now edited directly in Word (Rev0.5,
new text in red; PI decision of 28 October), so only the authors' Results record is generated here:

    Paper_A_SPICEhead/Brain Stimulation/v23/RESULTS_v23.tmpl.md  ->  v23/RESULTS_v23.md

Markers as in render_v22: {{key}} / {{key|fmt}} (a computed value), {{table:name}} (a table built in values_v23)
and {{missing_list}}.

  SPICEHEAD_MODEL=v23 python pipeline/render_v23.py           draft: a pending value renders as [PENDING key]
  SPICEHEAD_MODEL=v23 python pipeline/render_v23.py --final   refuse while any value is pending or any input missing
"""
import io
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import values_v23 as VV                                                # noqa: E402  (refuses unless v23)
from render_v22 import MARKER, PENDING, fmt, BS                        # noqa: E402

V23 = os.path.join(BS, "v23")
RTMPL = os.path.join(V23, "RESULTS_v23.tmpl.md")
ROUT = os.path.join(V23, "RESULTS_v23.md")


def render(final=False):
    V, T = VV.values()
    text = io.open(RTMPL, encoding="utf-8").read()
    if final and "<!--CHECK" in text:
        raise SystemExit("unresolved <!--CHECK--> notes in the template")
    pending = []

    def sub(m):
        key, spec = m.group(1).strip(), m.group(2)
        if key.startswith("table:"):
            if key[6:] in T:
                return T[key[6:]]
            pending.append(key)
            return PENDING.format(key)
        v = V.get(key)
        if v is None or (isinstance(v, float) and v != v):
            pending.append(key)
            return PENDING.format(key)
        return fmt(v, spec)

    # a computational status, not scientific finality
    miss = "\n".join(f"- {m}" for m in V["missing"]) or (
        "None: all inputs required by this results renderer are present. This is a computational status "
        "only; the v23 results figures and the Rev0.5 manuscript are separate steps.")
    text = MARKER.sub(sub, text.replace("{{missing_list}}", miss))
    if final and (pending or V["missing"]):
        raise SystemExit("not final: pending " + ", ".join(sorted(set(pending)))
                         + ("; missing inputs: " + "; ".join(V["missing"]) if V["missing"] else ""))
    io.open(ROUT, "w", encoding="utf-8", newline="\n").write(text)
    return dict(pending=sorted(set(pending)), missing=V["missing"])


if __name__ == "__main__":
    r = render(final="--final" in sys.argv)
    print(f"wrote {ROUT}")
    if r["pending"]:
        print(f"{len(r['pending'])} pending values: " + ", ".join(r["pending"]))
    if r["missing"]:
        print("missing inputs:\n  - " + "\n  - ".join(r["missing"]))
