# -*- coding: utf-8 -*-
"""Report the formatting of a .docx: page setup, styles, fonts, spacing, and document skeleton.

Used to copy the house formatting of an existing accepted manuscript onto this submission set.

  python pipeline/inspect_docx_format.py <file.docx> [--skeleton]
"""
import re
import sys
import zipfile
from collections import Counter

import docx
from docx.shared import Pt

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def pt(v):
    return None if v is None else round(v / 12700, 1)


def main(path, skeleton=False):
    d = docx.Document(path)
    print(f"=== {path} ===\n")

    # ---- page setup ----
    s = d.sections[0]
    inch = 914400
    print("PAGE SETUP")
    print(f"  page      {s.page_width / inch:.2f} x {s.page_height / inch:.2f} in "
          f"({'A4' if abs(s.page_width - 7560000) < 60000 else 'Letter/other'})")
    print(f"  margins   top {s.top_margin / inch:.2f}  bottom {s.bottom_margin / inch:.2f}  "
          f"left {s.left_margin / inch:.2f}  right {s.right_margin / inch:.2f} in")

    # line numbering and columns come from the sectPr
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
        styles_xml = z.read("word/styles.xml").decode("utf-8")
    ln = re.search(r"<w:lnNumType[^/]*/>", xml)
    print(f"  line numbers  {'yes: ' + ln.group(0) if ln else 'no'}")
    cols = re.search(r'<w:cols[^/]*w:num="(\d+)"', xml)
    print(f"  columns   {cols.group(1) if cols else 1}")

    # ---- default (docDefaults) ----
    m = re.search(r"<w:docDefaults>.*?</w:docDefaults>", styles_xml, re.S)
    if m:
        blk = m.group(0)
        f = re.search(r'<w:rFonts[^>]*w:ascii="([^"]+)"', blk)
        sz = re.search(r'<w:sz w:val="(\d+)"', blk)
        sp = re.search(r'<w:spacing([^/]*)/>', blk)
        print("\nDOCUMENT DEFAULTS")
        print(f"  font      {f.group(1) if f else '?'}  {int(sz.group(1)) / 2 if sz else '?'} pt")
        print(f"  spacing   {sp.group(1).strip() if sp else 'default'}")

    # ---- named styles actually used ----
    used = Counter(p.style.name for p in d.paragraphs)
    print("\nPARAGRAPH STYLES IN USE")
    for name, n in used.most_common():
        try:
            st = d.styles[name]
        except KeyError:
            print(f"  {name:26s} x{n}")
            continue
        fnt, pf = st.font, st.paragraph_format
        bits = []
        if fnt.name:
            bits.append(fnt.name)
        if fnt.size:
            bits.append(f"{fnt.size.pt:g}pt")
        if fnt.bold:
            bits.append("bold")
        if fnt.italic:
            bits.append("italic")
        if pf.space_before is not None:
            bits.append(f"before {pf.space_before.pt:g}")
        if pf.space_after is not None:
            bits.append(f"after {pf.space_after.pt:g}")
        if pf.line_spacing is not None:
            bits.append(f"line {pf.line_spacing}")
        if pf.alignment is not None:
            bits.append(str(pf.alignment).split()[0].lower())
        if pf.first_line_indent is not None:
            bits.append(f"indent {pf.first_line_indent.pt:g}")
        print(f"  {name:26s} x{n:<4} {', '.join(bits) or '(inherits)'}")

    # ---- direct run formatting, sampled ----
    fonts, sizes = Counter(), Counter()
    for p in d.paragraphs:
        for r in p.runs:
            if r.font.name:
                fonts[r.font.name] += 1
            if r.font.size:
                sizes[r.font.size.pt] += 1
    print(f"\nDIRECT RUN FONTS   {dict(fonts.most_common(5))}")
    print(f"DIRECT RUN SIZES   {dict(sizes.most_common(5))}")

    # ---- tables ----
    print(f"\nTABLES  {len(d.tables)}")
    for i, t in enumerate(d.tables[:3], 1):
        print(f"  table {i}: style={t.style.name if t.style else None}  "
              f"{len(t.rows)}x{len(t.columns)}")

    # ---- fields ----
    instr = re.findall(r"<w:instrText[^>]*>(.*?)</w:instrText>", xml, re.S)
    kinds = Counter(i.strip().split()[1] if len(i.strip().split()) > 1 else i.strip()[:20]
                    for i in instr if i.strip().startswith("ADDIN"))
    print(f"\nFIELDS  {len(instr)} total  {dict(kinds)}")

    if skeleton:
        print("\nDOCUMENT SKELETON")
        for p in d.paragraphs:
            t = p.text.strip()
            if not t:
                continue
            if p.style.name.startswith("Heading") or p.style.name == "Title":
                print(f"  [{p.style.name:10s}] {t[:88]}")
            elif len(t) < 90 and (t.isupper() or t.endswith(":")):
                print(f"  [{p.style.name:10s}] {t[:88]}")


if __name__ == "__main__":
    main(sys.argv[1], "--skeleton" in sys.argv)
