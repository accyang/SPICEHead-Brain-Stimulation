# -*- coding: utf-8 -*-
"""Helpers for revising the author's Word files in place, with new text in red.

Unchanged words keep their run formatting (including the colors of earlier revisions); replaced or inserted words
are red; deleted words are dropped and recorded in a change log. Used by build_rev04.py.
"""
import copy
import re
from difflib import SequenceMatcher

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.text.paragraph import Paragraph

RED = RGBColor(0xFF, 0x00, 0x00)
LOG = []


def _tokens(s):
    return re.findall(r"\S+\s*", s)


def set_text(p, new, label="", whole_threshold=0.45):
    """Rewrite paragraph p to `new`. Words kept from the old text keep their formatting; new words are red. When less
    than `whole_threshold` of the words match, the whole paragraph is marked new (avoids confetti marking)."""
    runs = list(p.runs)
    old = "".join(r.text for r in runs)
    if old == new:
        return p
    rprs = [r._r.rPr for r in runs]
    owner = [i for i, r in enumerate(runs) for _ in r.text]
    ot, nt = _tokens(old), _tokens(new)
    starts, c = [], len(old) - len(old.lstrip())
    for t in ot:
        starts.append(c)
        c += len(t)
    sm = SequenceMatcher(None, [t.strip() for t in ot], [t.strip() for t in nt], autojunk=False)
    allnew = not runs or sm.ratio() < whole_threshold
    segs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            continue
        ref = owner[starts[min(i1, len(ot) - 1)]] if ot else 0
        for k in range(j1, j2):
            if tag == "equal" and not allnew:
                segs.append([nt[k], owner[starts[i1 + k - j1]], False])
            else:
                segs.append([nt[k], ref, True])
    # an island of one or two kept words between new text reads as noise: mark it new
    i = 0
    while i < len(segs):
        if not segs[i][2]:
            j = i
            while j < len(segs) and not segs[j][2]:
                j += 1
            if j - i <= 2 and i > 0 and j < len(segs):
                for k in range(i, j):
                    segs[k][2] = True
            i = j
        else:
            i += 1
    for el in list(p._p):
        if el.tag != qn("w:pPr"):
            p._p.remove(el)
    groups = []
    for s in segs:
        if groups and groups[-1][1] == s[1] and groups[-1][2] == s[2]:
            groups[-1][0] += s[0]
        else:
            groups.append(list(s))
    for text, ref, isnew in groups:
        r = p.add_run(text)
        if runs and rprs[ref] is not None:
            r._r.insert(0, copy.deepcopy(rprs[ref]))
        if isnew:
            r.font.color.rgb = RED
    LOG.append(dict(label=label, before=old, after=new, similarity=round(sm.ratio(), 3), whole=allnew))
    return p


def replace_all(p, new, label=""):
    """Whole paragraph new (red), keeping the first run's formatting."""
    rpr = p.runs[0]._r.rPr if p.runs else None
    old = p.text
    for el in list(p._p):
        if el.tag != qn("w:pPr"):
            p._p.remove(el)
    r = p.add_run(new)
    if rpr is not None:
        r._r.insert(0, copy.deepcopy(rpr))
    r.font.color.rgb = RED
    LOG.append(dict(label=label, before=old, after=new, similarity=None, whole=True))
    return p


def delete(p, label=""):
    LOG.append(dict(label=label, before=p.text, after=None))
    p._p.getparent().remove(p._p)


def insert_after(anchor, text, like, label=""):
    """New red paragraph after `anchor` (a Paragraph or table element), formatted like paragraph `like`."""
    el = anchor._p if hasattr(anchor, "_p") else anchor
    new = copy.deepcopy(like._p)
    for ch in list(new):
        if ch.tag != qn("w:pPr"):
            new.remove(ch)
    el.addnext(new)
    p = Paragraph(new, like._parent)
    r = p.add_run(text)
    if like.runs and like.runs[0]._r.rPr is not None:
        r._r.insert(0, copy.deepcopy(like.runs[0]._r.rPr))
    r.font.color.rgb = RED
    LOG.append(dict(label=label, before=None, after=text))
    return p


def new_table(doc, after_el, headers, body, old=None, size=9, widths=None):
    """A red 'Table Grid' table after `after_el`, replacing table `old` (removed) when given."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = old.style if old is not None and old.style is not None else "Table Grid"
    for c, s in zip(t.rows[0].cells, headers):
        c.text = str(s)
    for row in body:
        for c, s in zip(t.add_row().cells, row):
            c.text = str(s)
    for i, row in enumerate(t.rows):
        pr = row._tr.get_or_add_trPr()
        pr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            pr.append(OxmlElement("w:tblHeader"))
        for c in row.cells:
            for p in c.paragraphs:
                # tables are single-spaced, whatever the document's body spacing
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.first_line_indent = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.color.rgb = RED
                    if i == 0:
                        r.font.bold = True
    if widths:
        from docx.shared import Cm
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    after_el.addnext(t._tbl)
    if old is not None:
        old._tbl.getparent().remove(old._tbl)
    LOG.append(dict(label="table", before=None if old is None else "replaced", after=" | ".join(map(str, headers))))
    return t
