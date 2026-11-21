# -*- coding: utf-8 -*-
"""The house format taken from NMH_Analysis_manuscript_260808.docx.

Read off the reference file rather than guessed:

  page          A4 (11909 x 16834 twips), 1 inch margins on all four sides
  body text     Times New Roman 12 pt, double spaced (line 480 twentieths), no space after
  line numbers  continuous, every line, 0.25 inch from the text
  headings      no Word Heading styles at all. Every paragraph is Normal; headings are bold
                Normal text, and the title is bold 14 pt
  tables        Table Grid
  references    Bibliography style, hanging indent of 384 twips
  Zotero prefs  a docVar named ZOTERO_PREF_1 in settings.xml, which is where Zotero 7 keeps them,
                rather than a field in the body

Applied by md_to_docx.py when MD2DOCX_HOUSE=nmh, and by zotero_fields.py.
"""
import re
import zipfile

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Twips

PAGE_W, PAGE_H = 11909, 16834          # A4, twips
MARGIN = 1440                          # 1 inch
LINE = 480                             # double spacing for 12 pt
FONT, SIZE = "Times New Roman", 12
TITLE_SIZE = 14
BIB_HANG = 384


def apply_document(doc):
    """Page setup, Normal, and the Bibliography style."""
    for s in doc.sections:
        s.page_width, s.page_height = Twips(PAGE_W), Twips(PAGE_H)
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Twips(MARGIN)

    n = doc.styles["Normal"]
    n.font.name = FONT
    n.font.size = Pt(SIZE)
    pf = n.paragraph_format
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)
    pf.line_spacing = 2.0

    # east-asian font has to be set on the element or Word substitutes for CJK runs
    rpr = n.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)

    try:
        b = doc.styles["Bibliography"]
    except KeyError:
        b = None
    if b is not None:
        bpf = b.paragraph_format
        bpf.left_indent = Twips(BIB_HANG)
        bpf.first_line_indent = Twips(-BIB_HANG)


def qn(tag):
    from docx.oxml.ns import qn as _qn
    return _qn(tag)


def heading(doc, level, title=False):
    """A heading the NMH way: a Normal paragraph whose runs are bold."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level <= 2 else 6)
    p.paragraph_format.space_after = Pt(0)
    return p


def style_heading_runs(par, level, title=False):
    for r in par.runs:
        r.bold = True
        r.font.size = Pt(TITLE_SIZE if title else SIZE)


def add_line_numbers(docx_path):
    """Continuous line numbering, which python-docx cannot set directly."""
    with zipfile.ZipFile(docx_path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    xml = parts["word/document.xml"].decode("utf-8")
    if "<w:lnNumType" not in xml:
        xml = xml.replace(
            "<w:cols",
            '<w:lnNumType w:countBy="1" w:distance="360" w:restart="continuous"/><w:cols', 1)
        parts["word/document.xml"] = xml.encode("utf-8")
        with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
            for k, v in parts.items():
                z.writestr(k, v)
    return True


def set_zotero_pref_docvar(docx_path, prefs_xml):
    """Store Zotero document preferences where Zotero 7 keeps them, in settings.xml docVars."""
    with zipfile.ZipFile(docx_path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    stg = parts.get("word/settings.xml", b"").decode("utf-8")
    if not stg:
        return False
    esc = (prefs_xml.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
           .replace('"', "&quot;"))
    dv = f'<w:docVars><w:docVar w:name="ZOTERO_PREF_1" w:val="{esc}"/></w:docVars>'
    stg = re.sub(r"<w:docVars>.*?</w:docVars>", "", stg, flags=re.S)
    # docVars must precede rsids in the schema order Word accepts
    if "<w:rsids>" in stg:
        stg = stg.replace("<w:rsids>", dv + "<w:rsids>", 1)
    else:
        stg = stg.replace("</w:settings>", dv + "</w:settings>", 1)
    parts["word/settings.xml"] = stg.encode("utf-8")
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in parts.items():
            z.writestr(k, v)
    return True
