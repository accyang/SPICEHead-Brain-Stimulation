# -*- coding: utf-8 -*-
"""Render the v22 methods paper from its template and values_v22.

The template `Paper_A_SPICEhead/Brain Stimulation/v22/04_manuscript_v22.tmpl.md` carries three kinds of marker:

    {{key}} / {{key|fmt}}   a computed value (fmt: pct, pct2, sci, num, numN, or a format spec)
    {{table:name}}          a table built in values_v22, or the renumbered reference list
    [@n,m–k]                a citation by its number in references_vancouver.md (1-37) or
                            v22/references_extra.md (38-); renumbered here in order of first
                            citation, so the published list is always in citation order

and two computed after substitution: {{wc.abstract}} and {{wc.body}}, measured with the audit's
rules (pipeline/wordcount.py for the body).

  python pipeline/render_v22.py            draft: a pending value renders as [PENDING key]
  python pipeline/render_v22.py --final    refuse while any value is pending or unresolved
  python pipeline/render_v22.py --docx     also build the .docx files (house style)
  python pipeline/render_v22.py --zotero   also build 04_manuscript_v22_zotero_fields.docx (live citations)
  python pipeline/render_v22.py --outdir D write to D instead (test renders, e.g. on v21 data)

Outputs in Paper_A_SPICEhead/Brain Stimulation/v22/: 04_manuscript_v22.md (working draft, keeps ⚠️ blocks and the
DRAFT banner) and 04_manuscript_v22_submission.md (stripped by strip_scaffolding.strip), plus
their .docx with --docx. Never written: any Word file outside v22/, in particular the live
04_manuscript_zotero_fields_0827.docx.
"""
import io
import os
import re
import sys
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, _here)
os.environ.setdefault("SPICEHEAD_MODEL", "v22")   # v22 record script; the package default is v23 since 2 Nov 2026
import values_v22 as VV                                                # noqa: E402

BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
V22 = os.path.join(BS, "v22")
TMPL = os.path.join(V22, "04_manuscript_v22.tmpl.md")
OUT = os.path.join(V22, "04_manuscript_v22.md")
SUB = os.path.join(V22, "04_manuscript_v22_submission.md")
STMPL = os.path.join(V22, "06_supplement_v22.tmpl.md")
SOUT = os.path.join(V22, "06_supplement_v22.md")
SSUB = os.path.join(V22, "06_supplement_v22_submission.md")
RTMPL = os.path.join(V22, "RESULTS_v22.tmpl.md")
ROUT = os.path.join(V22, "RESULTS_v22.md")
REFS = [os.path.join(BS, "references_vancouver.md"), os.path.join(V22, "references_extra.md")]
MARKER = re.compile(r"\{\{([^}|]+?)(?:\|([^}]+))?\}\}")
CITE = re.compile(r"\[@([\d,–\s]+)\]")
PENDING = "[PENDING {}]"


def fmt(v, spec):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if spec in (None, ""):
        if isinstance(v, float):
            return f"{v:g}" if abs(v) >= 1e-3 or v == 0 else VV.sci(v)
        if isinstance(v, list):
            return ", ".join(fmt(x, None) for x in v)
        return str(v)
    if spec == "pct":
        return VV.pct(v)
    if spec == "pct2":
        return VV.pct(v, 2)
    if spec == "sci":
        return VV.sci(v)
    if spec.startswith("num"):
        return VV.num(v, int(spec[3:]) if len(spec) > 3 else 3)
    out = format(v, spec)
    return out.replace("-", VV.MINUS) if out.startswith("-") else out


def expand(group):
    nums = []
    for part in group.replace(" ", "").split(","):
        if "–" in part:
            a, b = part.split("–")
            nums += list(range(int(a), int(b) + 1))
        else:
            nums.append(int(part))
    return nums


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i]}–{nums[j]}")
        else:
            out += [str(n) for n in nums[i:j + 1]]
        i = j + 1
    return ",".join(out)


def references():
    ref = {}
    for p in REFS:
        for ln in io.open(p, encoding="utf-8").read().splitlines():
            m = re.match(r"^\[(\d+)\]\s+(.*)$", ln.strip())
            if m:
                ref[int(m.group(1))] = m.group(2)
    return ref


def renumber(text, ref):
    order = {}
    for m in CITE.finditer(text):
        for n in expand(m.group(1)):
            if n not in ref:
                raise SystemExit(f"citation [{n}] is in neither reference file")
            order.setdefault(n, len(order) + 1)
    body = CITE.sub(lambda m: "[" + compress([order[n] for n in expand(m.group(1))]) + "]", text)
    lst = "\n\n".join(f"[{new}] {ref[old]}" for old, new in sorted(order.items(), key=lambda x: x[1]))
    return body, lst, order


def write_reference_files(vancouver, order, outdir):
    """The renumbered list, as Vancouver text and as CSL-JSON, for zotero_fields.py."""
    import json
    io.open(os.path.join(outdir, "references_v22_vancouver.md"), "w", encoding="utf-8",
            newline="\n").write(vancouver + "\n")
    csl = {}
    for p in (os.path.join(BS, "references.csl.json"), os.path.join(V22, "references_extra.csl.json")):
        for e in json.load(open(p, encoding="utf-8")):
            csl[int(e["citation-number"])] = e
    out = []
    for old, new in sorted(order.items(), key=lambda x: x[1]):
        if old not in csl:
            raise SystemExit(f"reference [{old}] has no CSL record")
        e = dict(csl[old], **{"citation-number": new, "id": f"ref{new}"})
        out.append(e)
    json.dump(out, open(os.path.join(outdir, "references_v22.csl.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def clean(text):
    """Submission copies carry no notes to the authors (the working draft keeps them)."""
    text = re.sub(r"[ \t]*<!--.*?-->", "", text, flags=re.S)
    # author placeholders stay as neutral brackets, without the working-draft warning mark
    return re.sub(r"⚠️\s*(?=\[)", "", text)


def abstract_words(text):
    ab = text.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
    ab = re.sub(r"\*\*(Background|Objective|Methods|Results|Conclusions)\.\*\*|[✅⚠️]", "", ab)
    return len(ab.split())


def body_words(text):
    from wordcount import body_words as bw
    fd, p = tempfile.mkstemp(suffix=".md")
    os.close(fd)
    try:
        io.open(p, "w", encoding="utf-8").write(text)
        return bw(p)[0]
    finally:
        os.remove(p)


def render(final=False, outdir=None):
    global OUT, SUB
    if outdir:                         # test renders (e.g. against v21 data) never touch v22/
        OUT = os.path.join(outdir, os.path.basename(OUT))
        SUB = os.path.join(outdir, os.path.basename(SUB))
    V, T = VV.values()
    text = io.open(TMPL, encoding="utf-8").read()
    if final and "<!--CHECK" in text:
        raise SystemExit("unresolved <!--CHECK--> notes in the template")
    text, T["references"], order = renumber(text, references())
    write_reference_files(T["references"], order, os.path.dirname(OUT))
    pending = []

    def sub(m):
        key, spec = m.group(1).strip(), m.group(2)
        if key.startswith("table:"):
            name = key[6:]
            if name in T:
                return T[name]
            pending.append(key)
            return PENDING.format(key)
        if key.startswith("wc."):
            return m.group(0)                      # filled after the text is final
        v = V.get(key)
        if v is None or (isinstance(v, float) and v != v):
            pending.append(key)
            return PENDING.format(key)
        return fmt(v, spec)

    text = MARKER.sub(sub, text)
    counted = re.sub(r"<!--.*?-->", "", text, flags=re.S)      # notes to the authors are not prose
    na, nb = abstract_words(counted), body_words(counted)
    text = text.replace("{{wc.abstract}}", str(na)).replace("{{wc.body}}", f"{nb:,}")
    if final and (pending or V["missing"]):
        raise SystemExit("not final: pending " + ", ".join(sorted(set(pending)))
                         + ("; missing inputs: " + "; ".join(V["missing"]) if V["missing"] else ""))
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
    from strip_scaffolding import strip
    io.open(SUB, "w", encoding="utf-8", newline="\n").write(clean(strip(text)))

    # the supplement: the same values and tables, no citations and no word limits
    n_main = len(pending)
    stext = io.open(STMPL, encoding="utf-8").read()
    if final and "<!--CHECK" in stext:
        raise SystemExit("unresolved <!--CHECK--> notes in the supplement template")
    stext = MARKER.sub(sub, stext)
    if final and len(pending) > n_main:
        raise SystemExit("supplement not final: pending " + ", ".join(sorted(set(pending[n_main:]))))
    d = os.path.dirname(OUT)
    for path, t in ((SOUT, stext), (SSUB, clean(strip(stext)))):
        io.open(os.path.join(d, os.path.basename(path)), "w", encoding="utf-8", newline="\n").write(t)

    # the authors' organized results record: same values, plus what is still missing
    rtext = io.open(RTMPL, encoding="utf-8").read()
    # a computational status, not scientific finality (review of 2 October, R5.5)
    miss = "\n".join(f"- {m}" for m in V["missing"]) or (
        "None: all inputs required by this results renderer are present. This is a computational "
        "status only; author-supplied items (imaging details, ethics, funding, licence) remain "
        "placeholders in the manuscript.")
    rtext = MARKER.sub(sub, rtext.replace("{{missing_list}}", miss))
    io.open(os.path.join(d, os.path.basename(ROUT)), "w", encoding="utf-8", newline="\n").write(rtext)
    return dict(abstract=na, body=nb, pending=sorted(set(pending)), missing=V["missing"],
                n_refs=len(order))


if __name__ == "__main__":
    a = sys.argv
    r = render(final="--final" in a, outdir=a[a.index("--outdir") + 1] if "--outdir" in a else None)
    print(f"wrote {OUT} and {os.path.basename(SUB)}")
    print(f"abstract {r['abstract']} words (limit 250); body {r['body']} words (limit 4000); "
          f"{r['n_refs']} references")
    if r["pending"]:
        print(f"{len(r['pending'])} pending values: " + ", ".join(r["pending"]))
    if r["missing"]:
        print("missing inputs:\n  - " + "\n  - ".join(r["missing"]))
    if "--docx" in sys.argv:
        os.environ.setdefault("MD2DOCX_HOUSE", "nmh")
        from md_to_docx import convert
        d = os.path.dirname(OUT)
        for p in (OUT, SUB, os.path.join(d, os.path.basename(SOUT)),
                  os.path.join(d, os.path.basename(SSUB))):
            print("wrote", convert(p, p[:-3] + ".docx"))
    if "--zotero" in sys.argv:
        # the deliverable the authors edit: live Zotero citation fields, from the submission copy
        import subprocess
        d = os.path.dirname(OUT)
        rc = subprocess.run([sys.executable, os.path.join(_here, "zotero_fields.py"),
                             "--src", SUB, "--out", os.path.join(d, "04_manuscript_v22_zotero_fields.docx"),
                             "--csl", os.path.join(d, "references_v22.csl.json"),
                             "--refs", os.path.join(d, "references_v22_vancouver.md")]).returncode
        if rc:
            raise SystemExit("zotero_fields.py failed")
