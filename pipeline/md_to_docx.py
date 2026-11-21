# -*- coding: utf-8 -*-
"""Minimal Markdown -> .docx converter (no pandoc on this machine).

Handles exactly the constructs used in the submission package: ATX headings, paragraphs, bullet
and numbered lists, pipe tables, blockquotes, horizontal rules, fenced code, and inline
**bold** / *italic* / `code`. Anything else passes through as plain text.

Usage:  python pipeline/md_to_docx.py <in.md> [<out.docx>]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

import house_format as HF

INLINE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*\n]+?\*(?!\*)|`[^`\n]+?`|\[[^\]]+\]\([^)]+\))")

# zotero_fields.py marks each citation point with ⟦ZCITE:…⟧ and the bibliography with ⟦ZBIBL⟧,
# then swaps those runs for Word field codes in the packaged XML. The swap is only safe if each
# marker occupies a run by itself, so when MD2DOCX_EXTRA_INLINE is set they are split out here.
if os.environ.get("MD2DOCX_EXTRA_INLINE"):
    INLINE = re.compile(r"(⟦[^⟧]+⟧|" + INLINE.pattern[1:])


def add_inline(par, text):
    """Render inline markdown into runs on an existing paragraph."""
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            if os.environ.get("MD2DOCX_HOUSE") != "nmh":
                r.font.name = "Consolas"
                r.font.size = Pt(9)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        elif tok.startswith("[") and "](" in tok:
            label = tok[1:tok.index("](")]
            par.add_run(label).underline = True
        else:
            par.add_run(tok)


def flush_table(doc, rows):
    """rows: list of list[str]; row 1 is the markdown separator and is dropped."""
    body = [r for i, r in enumerate(rows) if not (i == 1 and all(set(c) <= set("-: ") for c in r))]
    if not body:
        return
    ncol = max(len(r) for r in body)
    t = doc.add_table(rows=len(body), cols=ncol)
    t.style = "Table Grid"
    for i, row in enumerate(body):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            add_inline(p, row[j] if j < len(row) else "")
            for r in p.runs:
                r.font.size = Pt(9)
                if i == 0:
                    r.bold = True


def convert(md_path, out_path=None):
    out_path = out_path or os.path.splitext(md_path)[0] + ".docx"
    lines = open(md_path, encoding="utf-8").read().split("\n")
    doc = docx.Document()
    house = os.environ.get("MD2DOCX_HOUSE") == "nmh"
    if house:
        HF.apply_document(doc)          # A4, 1" margins, Times New Roman 12, double spaced
    else:
        st = doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)

    tbl, code, i = [], False, 0
    while i < len(lines):
        ln = lines[i].rstrip()

        if ln.startswith("```"):
            code = not code
            i += 1
            continue
        if code:
            p = doc.add_paragraph()
            r = p.add_run(ln)
            r.font.name = "Consolas"
            r.font.size = Pt(9)
            i += 1
            continue

        # pipe table
        if ln.startswith("|"):
            tbl.append([c.strip() for c in ln.strip().strip("|").split("|")])
            i += 1
            if i >= len(lines) or not lines[i].rstrip().startswith("|"):
                flush_table(doc, tbl)
                tbl = []
                doc.add_paragraph()
            continue

        if not ln.strip():
            i += 1
            continue

        if re.match(r"^\s*(---+|\*\*\*+)\s*$", ln):
            doc.add_paragraph("_" * 60).alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = min(len(m.group(1)), 4)
            text = re.sub(r"<!--.*?-->", "", m.group(2)).strip()
            if house:
                # the reference manuscript uses no Word heading styles: every paragraph is
                # Normal and headings are simply bold, with the title at 14 pt
                h = HF.heading(doc, lvl, title=(lvl == 1))
                add_inline(h, text)
                HF.style_heading_runs(h, lvl, title=(lvl == 1))
            else:
                h = doc.add_heading(level=lvl)
                add_inline(h, text)
            i += 1
            continue

        if ln.lstrip().startswith(">"):
            body = ln.lstrip()[1:].strip()
            p = doc.add_paragraph(style="Intense Quote" if "Intense Quote" in
                                  [s.name for s in doc.styles] else None)
            add_inline(p, body)
            i += 1
            continue

        m = re.match(r"^\s*[-*+]\s+(.*)$", ln)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, m.group(1))
            i += 1
            continue

        m = re.match(r"^\s*\d+[.)]\s+(.*)$", ln)
        if m:
            p = doc.add_paragraph(style="List Number")
            add_inline(p, m.group(1))
            i += 1
            continue

        # paragraph: join soft-wrapped lines
        buf = [ln]
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not re.match(r"^\s*([-*+]\s|\d+[.)]\s|#{1,6}\s|\||>|```|---)", lines[i])):
            buf.append(lines[i].strip())
            i += 1
        text = " ".join(buf)
        if house and re.match(r"^\[\d+\]\s", text):
            try:
                p = doc.add_paragraph(style="Bibliography")
            except KeyError:
                p = doc.add_paragraph()
        else:
            p = doc.add_paragraph()
        add_inline(p, text)

    if tbl:
        flush_table(doc, tbl)
    doc.save(out_path)
    if house:
        # continuous line numbering lives in the sectPr, which python-docx cannot reach
        HF.add_line_numbers(out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print("wrote", convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
