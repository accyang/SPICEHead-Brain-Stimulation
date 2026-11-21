# -*- coding: utf-8 -*-
"""Produce Zotero-ready reference files and a citation-marker copy of the manuscript.

Two things are needed to move this bibliography into Zotero, and they are different:

  1. A library Zotero can import. Zotero reads CSL-JSON, RIS and BibTeX natively. The CSL-JSON
     and RIS already exist; this adds a BibTeX file with stable citation keys, and re-checks that
     all three agree on count and DOI set.

  2. A manuscript whose citations Zotero can *recognize*. Numeric "[12]" markers carry no author
     or year, so Zotero cannot match them to library items. Zotero's RTF/ODF Scan feature reads
     temporary citation markers of the form {Author, Year} and converts them into live citations.
     This writes 04_manuscript_zotero.md, identical to the submission copy except that every
     numeric citation is replaced by such a marker, ready for Tools > RTF Scan after conversion
     to .rtf or .odt.

  python pipeline/zotero_export.py
"""
import io
import json
import os
import re
import unicodedata

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")

BIBTEX_TYPE = {"article-journal": "article", "webpage": "misc", "chapter": "incollection",
               "paper-conference": "inproceedings", "book": "book"}


def ascii_key(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if c.isalnum() and ord(c) < 128)


def esc(s):
    return (s or "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def year_of(e):
    try:
        return e["issued"]["date-parts"][0][0]
    except (KeyError, IndexError, TypeError):
        return ""


def key_of(e):
    fam = ascii_key((e.get("author") or [{}])[0].get("family", "Anon")) or "Anon"
    word = ""
    for w in re.findall(r"[A-Za-z]+", e.get("title", "")):
        if len(w) > 3 and w.lower() not in ("with", "from", "that", "this", "into", "using"):
            word = ascii_key(w).lower()
            break
    return f"{fam.lower()}{year_of(e)}{word}"


def main():
    csl = json.load(open(os.path.join(BS, "references.csl.json"), encoding="utf-8"))
    csl.sort(key=lambda e: e["citation-number"])

    # ---- 1. BibTeX with stable keys -------------------------------------------------
    keys, lines, seen = {}, [], set()
    for e in csl:
        k = key_of(e)
        while k in seen:
            k += "a"
        seen.add(k)
        keys[e["citation-number"]] = k
        t = BIBTEX_TYPE.get(e.get("type", ""), "misc")
        f = [f"  author = {{{' and '.join(a.get('family','') + ', ' + a.get('given','') for a in e.get('author') or []) or 'Anon'}}}",
             f"  title = {{{esc(e.get('title'))}}}"]
        if e.get("container-title"):
            f.append(f"  journal = {{{esc(e['container-title'])}}}")
        if year_of(e):
            f.append(f"  year = {{{year_of(e)}}}")
        for tag, k2 in (("volume", "volume"), ("number", "issue"), ("pages", "page"),
                        ("publisher", "publisher"), ("doi", "DOI"), ("url", "URL")):
            if e.get(k2):
                f.append(f"  {tag} = {{{esc(str(e[k2]))}}}")
        lines.append("@" + t + "{" + k + ",\n" + ",\n".join(f) + "\n}")
    p = os.path.join(BS, "references.bib")
    io.open(p, "w", encoding="utf-8", newline="").write("\n\n".join(lines) + "\n")
    print(f"  wrote {os.path.relpath(p, ROOT)} ({len(lines)} entries)")

    # ---- 2. cross-format agreement ---------------------------------------------------
    ris = io.open(os.path.join(BS, "references.ris"), encoding="utf-8").read()
    van = io.open(os.path.join(BS, "references_vancouver.md"), encoding="utf-8").read()
    n_ris = ris.count("TY  -")
    n_van = len(re.findall(r"^\[\d+\]", van, re.M))
    doi_csl = {(e.get("DOI") or "").lower() for e in csl if e.get("DOI")}
    doi_ris = {m.lower() for m in re.findall(r"^DO  - (.+)$", ris, re.M)}
    ok = (n_ris == n_van == len(csl)) and doi_csl == doi_ris
    print(f"  formats agree: CSL {len(csl)}, RIS {n_ris}, Vancouver {n_van}, "
          f"BibTeX {len(lines)}; DOI sets {'match' if doi_csl == doi_ris else 'DIFFER'}")

    # ---- 3. Zotero RTF/ODF-Scan marker copy of the manuscript ------------------------
    marker = {}
    for e in csl:
        a = e.get("author") or []
        fam = (a[0].get("family") if a else "") or e.get("publisher") or "Anon"
        if len(a) == 2:
            name = f"{fam} & {a[1].get('family','')}"
        elif len(a) > 2:
            name = f"{fam} et al."
        else:
            name = fam
        marker[e["citation-number"]] = f"{{{name}, {year_of(e)}}}"

    src = os.path.join(BS, "04_manuscript_submission.md")
    if not os.path.exists(src):
        print("  (run strip_scaffolding.py first for the submission copy)")
        return 1
    doc = io.open(src, encoding="utf-8").read()
    body, _, refs = doc.partition("\n## References")

    def sub(m):
        nums = []
        for part in m.group(1).split(","):
            part = part.strip().replace("–", "-").replace("—", "-")
            a, _, b = part.partition("-")
            if a.isdigit() and b.isdigit():
                nums += list(range(int(a), int(b) + 1))
            elif part.isdigit():
                nums.append(int(part))
        if not nums or any(n not in marker for n in nums):
            return m.group(0)
        return "".join(marker[n] for n in nums)

    n_before = len(re.findall(r"\[(?:\d+)(?:\s*[,–—-]\s*\d+)*\]", body))
    body = re.sub(r"\[((?:\d+)(?:\s*[,–—-]\s*\d+)*)\]", sub, body)
    note = ("\n\n---\n\n*Citations in this copy are Zotero RTF/ODF-Scan temporary markers of the "
            "form {Author, Year}. Import references.bib, references.ris or references.csl.json "
            "into Zotero, save this file as .rtf or .odt, then run Tools > RTF Scan to convert "
            "the markers into live citations. The numbered reference list below is retained for "
            "checking and should be deleted once the scan has run.*\n")
    out = os.path.join(BS, "04_manuscript_zotero.md")
    io.open(out, "w", encoding="utf-8", newline="").write(body + note + "\n## References" + refs)
    print(f"  wrote {os.path.relpath(out, ROOT)} ({n_before} numeric citations converted)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
