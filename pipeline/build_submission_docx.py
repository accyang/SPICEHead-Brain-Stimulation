# -*- coding: utf-8 -*-
"""Word files of the Brain Stimulation submission package, with the figures embedded.

Converts `04_manuscript.md` and `06_supplement.md` in `Paper_A_SPICEhead/Brain Stimulation/submission_2026-10-04/`
to .docx with the project's converter (pipeline/md_to_docx.py, house format: A4, Times New Roman
12 pt, double spacing, continuous line numbers), then places each figure, on a new page, directly
above its legend in the legend section ("Figure captions" in the manuscript, "Supplementary
figure legends" in the supplement). The text is not edited: the converter renders the Markdown
as written and the only additions are the picture paragraphs.

Figures come from `figures/` (pipeline/make_submission_figures.py); for embedding they are
resampled to about 380 dpi at the 6.27-inch text width so the Word files stay small. The PDF and
600 dpi PNG originals remain the artwork files for upload.

  python pipeline/build_submission_docx.py
"""
import io
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, _here)
os.environ["MD2DOCX_HOUSE"] = "nmh"
import md_to_docx                                                     # noqa: E402

import docx                                                           # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK             # noqa: E402
from docx.shared import Inches                                       # noqa: E402
from PIL import Image                                                # noqa: E402

PKG = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation", "submission_2026-10-04")
FIG = os.path.join(PKG, "figures")
TEXT_W = 6.27                       # inches: A4 with 1-inch margins
MAX_H = 7.6                         # leave room for the legend's first lines on the same page
EMBED_PX = 2400

DOCS = (("04_manuscript.md", "04_manuscript.docx", "Figure captions",
         [("Figure 1.", "Figure1"), ("Figure 2.", "Figure2"), ("Figure 3.", "Figure3"),
          ("Figure 4.", "Figure4"), ("Figure 5.", "Figure5")]),
        ("06_supplement.md", "06_supplement.docx", "Supplementary figure legends",
         [("Figure S1.", "FigureS1"), ("Figure S2.", "FigureS2"), ("Figure S3.", "FigureS3")]))


def embed_copy(name):
    """A lighter PNG for embedding (the 600 dpi file stays the artwork original)."""
    src = os.path.join(FIG, f"{name}.png")
    im = Image.open(src)
    if im.width > EMBED_PX:
        im = im.resize((EMBED_PX, round(im.height * EMBED_PX / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf, im.width, im.height


TMP = os.environ.get("SPICEHEAD_DOCX_TMP", r"C:\spicehead_v22_logs\docx_build")


def unwrap(md_path):
    """A copy of the Markdown in which a soft-wrapped line that happens to begin with "N. " is
    joined to the paragraph it continues, so the converter does not start a numbered list in
    mid-sentence (e.g. "... thread count set to / 1. At the start ..."). Only line breaks change;
    every character of text is kept. The package files contain no intended numbered list that
    directly follows paragraph text without a blank line."""
    lines = open(md_path, encoding="utf-8").read().split("\n")
    out = []
    for ln in lines:
        prev = out[-1] if out else ""
        if (re.match(r"^\d+[.)]\s", ln) and prev.strip()
                and not re.match(r"^\s*([-*+]\s|\d+[.)]\s|#{1,6}\s|\||>|```)", prev)):
            out[-1] = prev.rstrip() + " " + ln.strip()
        else:
            out.append(ln)
    tmp = os.path.join(TMP, os.path.basename(md_path))
    open(tmp, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    return tmp


def build(md, out, heading, figs):
    """Build in a local folder (Dropbox locks a freshly written file while it syncs, which breaks
    the in-place rewrites), then copy the finished file into the package."""
    import shutil
    import time
    os.makedirs(TMP, exist_ok=True)
    md_path, out_path = os.path.join(PKG, md), os.path.join(TMP, out)
    md_to_docx.convert(unwrap(md_path), out_path)
    d = docx.Document(out_path)
    paras = d.paragraphs
    start = next(i for i, p in enumerate(paras) if p.text.strip() == heading)
    placed = []
    for label, name in figs:
        p = next((q for q in paras[start + 1:] if q.text.strip().startswith(label)), None)
        if p is None:
            raise SystemExit(f"{out}: legend '{label}' not found after '{heading}'")
        buf, w, h = embed_copy(name)
        width = TEXT_W if TEXT_W * h / w <= MAX_H else MAX_H * w / h
        pic = p.insert_paragraph_before()
        pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic.paragraph_format.line_spacing = 1.0
        pic.paragraph_format.keep_with_next = True
        pic.paragraph_format.page_break_before = True
        pic.add_run().add_picture(buf, width=Inches(width))
        placed.append((label, name, round(width, 2)))
    d.save(out_path)
    md_to_docx_house_line_numbers(out_path)
    final = os.path.join(PKG, out)
    for attempt in range(10):
        try:
            shutil.copyfile(out_path, final)
            break
        except OSError:
            time.sleep(3)
    else:
        raise SystemExit(f"could not write {final} (locked?)")
    return final, placed


def md_to_docx_house_line_numbers(path):
    """Saving with python-docx keeps the section properties, but re-apply the house line
    numbering in case the save dropped it (it lives in sectPr, which python-docx cannot reach)."""
    import house_format as HF
    HF.add_line_numbers(path)


def verify(md, out):
    """Every Markdown paragraph's words appear, in order, in the Word text (content unchanged)."""
    raw = open(os.path.join(PKG, md), encoding="utf-8").read()
    text = re.sub(r"[*`#>|]", " ", raw)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # links render as their text
    md_words = re.findall(r"[A-Za-z0-9]+", text)
    d = docx.Document(os.path.join(PKG, out))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    dw = re.findall(r"[A-Za-z0-9]+", " ".join(parts))
    from collections import Counter
    missing = Counter(md_words) - Counter(dw)
    n_pics = sum(len(p._element.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"))
                 for p in d.paragraphs)
    return len(md_words), sum(missing.values()), missing.most_common(8), n_pics


if __name__ == "__main__":
    for md, out, heading, figs in DOCS:
        path, placed = build(md, out, heading, figs)
        n, miss, top, pics = verify(md, out)
        print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
        for label, name, w in placed:
            print(f"   {label:12s} <- figures/{name}.png at {w} in")
        print(f"   words in Markdown {n}; not found in Word {miss} {top if miss else ''}; pictures {pics}")
