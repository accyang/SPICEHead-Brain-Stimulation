# -*- coding: utf-8 -*-
"""Renumber the bibliography into first-appearance order.

Elsevier's numbered style (`elsarticle-num`) requires reference [1] to be the first one cited,
[2] the second, and so on. The 4,000-word trim moved and removed text, and restoring the citations
it had orphaned left the numbering out of order in four places.

This recomputes the mapping from the manuscript template, then rewrites, consistently:
  * the citation brackets in the template (ranges are expanded, remapped, and re-collapsed)
  * references.csl.json  (reordered, `id` and `citation-number` renumbered)
  * references_vancouver.md
  * references.ris
  * doi_audit.md
  * the one place the response letter names a reference by number

It is idempotent: run it again and it reports that the order is already correct.

  python pipeline/renumber_refs.py [--check]
"""
import io
import json
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
TMPL = os.path.join(BS, "04_manuscript.tmpl.md")
CITE = re.compile(r"\[((?:\d+)(?:\s*[,–—-]\s*\d+)*)\]")


def parse(group):
    """'22–24,30' -> [22, 23, 24, 30]"""
    out = []
    for part in group.split(","):
        part = part.strip().replace("–", "-").replace("—", "-")
        a, _, b = part.partition("-")
        if a.strip().isdigit() and b.strip().isdigit():
            out.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def collapse(nums):
    """[22, 23, 24, 30] -> '22–24,30'"""
    nums = sorted(set(nums))
    runs, cur = [], [nums[0]]
    for n in nums[1:]:
        if n == cur[-1] + 1:
            cur.append(n)
        else:
            runs.append(cur)
            cur = [n]
    runs.append(cur)
    return ",".join(f"{r[0]}–{r[-1]}" if len(r) > 2 else
                    ",".join(str(x) for x in r) for r in runs)


def appearance_order(text):
    body = text[text.find("## 1. Introduction"):]
    order, seen = [], set()
    for m in CITE.finditer(body):
        for n in parse(m.group(1)):
            if n not in seen:
                seen.add(n)
                order.append(n)
    return order


def main(check=False):
    tmpl = io.open(TMPL, encoding="utf-8").read()
    order = appearance_order(tmpl)
    csl = json.load(open(os.path.join(BS, "references.csl.json"), encoding="utf-8"))

    missing = sorted(set(range(1, len(csl) + 1)) - set(order))
    if missing:
        raise SystemExit(f"cannot renumber: references never cited in the body: {missing}")

    if order == sorted(order):
        print(f"  ok    citation numbering is already in first-appearance order ({len(order)} refs)")
        return 0
    if check:
        bad = [(order[i], order[i + 1]) for i in range(len(order) - 1) if order[i] > order[i + 1]]
        print(f"  FAIL  citation numbering is not in first-appearance order; breaks at {bad}")
        return 1

    old2new = {old: i + 1 for i, old in enumerate(order)}
    print(f"  remapping {len(old2new)} references into citation order")
    for old, new in sorted(old2new.items()):
        if old != new:
            print(f"    [{old}] -> [{new}]")

    # ---- manuscript template -------------------------------------------------
    head, body = tmpl[:tmpl.find("## 1. Introduction")], tmpl[tmpl.find("## 1. Introduction"):]
    body = CITE.sub(lambda m: "[" + collapse([old2new[n] for n in parse(m.group(1))]) + "]", body)
    io.open(TMPL, "w", encoding="utf-8", newline="").write(head + body)
    print(f"  wrote {os.path.relpath(TMPL, ROOT)}")

    # ---- CSL-JSON ------------------------------------------------------------
    by_old = {int(e["citation-number"]): e for e in csl}
    out = []
    for old in order:
        e = dict(by_old[old])
        e["citation-number"] = old2new[old]
        e["id"] = f"ref{old2new[old]}"
        out.append(e)
    p = os.path.join(BS, "references.csl.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"  wrote {os.path.relpath(p, ROOT)}")

    # ---- Vancouver list ------------------------------------------------------
    p = os.path.join(BS, "references_vancouver.md")
    s = io.open(p, encoding="utf-8").read()
    entries = {int(m.group(1)): m.group(0)
               for m in re.finditer(r"^\[(\d+)\][^\n]*", s, re.M)}
    lines = []
    for old in order:
        lines.append(re.sub(r"^\[\d+\]", f"[{old2new[old]}]", entries[old]))
    io.open(p, "w", encoding="utf-8", newline="").write("\n\n".join(lines) + "\n")
    print(f"  wrote {os.path.relpath(p, ROOT)}")

    # ---- RIS (DOI-keyed, so reorder by the new sequence) ---------------------
    p = os.path.join(BS, "references.ris")
    s = io.open(p, encoding="utf-8").read()
    recs = [r.strip() for r in re.split(r"\nER  - *\n?", s) if r.strip()]
    # Key by DOI where there is one, and by title otherwise — the COMSOL manual has no DOI, and
    # a DOI-only index silently dropped it the first time this ran.
    def _key(rec_or_entry, is_rec):
        if is_rec:
            m = re.search(r"^DO  - (.+)$", rec_or_entry, re.M)
            if m:
                return ("doi", m.group(1).strip().lower())
            t = re.search(r"^TI  - (.+)$", rec_or_entry, re.M)
            return ("ti", t.group(1).strip().lower() if t else rec_or_entry[:60].lower())
        if rec_or_entry.get("DOI"):
            return ("doi", rec_or_entry["DOI"].lower())
        return ("ti", (rec_or_entry.get("title") or "").strip().lower())

    index = {_key(r, True): r for r in recs}
    ordered, leftover = [], dict(index)
    for e in out:
        k = _key(e, False)
        if k in leftover:
            ordered.append(leftover.pop(k))
    ordered += list(leftover.values())        # anything unmatched keeps its relative order
    if len(ordered) != len(recs):
        raise SystemExit(f"RIS reorder lost records: {len(recs)} in, {len(ordered)} out")
    io.open(p, "w", encoding="utf-8", newline="").write(
        "".join(r + "\nER  - \n\n" for r in ordered).rstrip() + "\n")
    print(f"  wrote {os.path.relpath(p, ROOT)} ({len(ordered)} entries)")

    # ---- DOI audit table -----------------------------------------------------
    p = os.path.join(BS, "doi_audit.md")
    if os.path.exists(p):
        s = io.open(p, encoding="utf-8").read()
        rows = {int(m.group(1)): m.group(0)
                for m in re.finditer(r"^\| (\d+) \|[^\n]*", s, re.M)}
        if rows:
            new_rows = [re.sub(r"^\| \d+ \|", f"| {old2new[old]} |", rows[old])
                        for old in order if old in rows]
            first = min(m.start() for m in re.finditer(r"^\| \d+ \|", s, re.M))
            last = max(m.end() for m in re.finditer(r"^\| \d+ \|[^\n]*", s, re.M))
            s = s[:first] + "\n".join(new_rows) + s[last:]
            s = re.sub(r"^\d+ of \d+ references resolved\.$",
                       f"{len(order)} of {len(order)} references resolved.", s, flags=re.M)
            io.open(p, "w", encoding="utf-8", newline="").write(s)
            print(f"  wrote {os.path.relpath(p, ROOT)}")

    # ---- the letter names one reference by number ----------------------------
    p = os.path.join(BS, "05_response_to_review.md")
    s = io.open(p, encoding="utf-8").read()
    lotes_old = next((int(e["citation-number"]) for e in csl
                      if e.get("DOI") == "10.1016/j.brs.2023.05.008"), None)
    if lotes_old and f"ref. [{lotes_old}]" in s:
        s = s.replace(f"ref. [{lotes_old}]", f"ref. [{old2new[lotes_old]}]")
        io.open(p, "w", encoding="utf-8", newline="").write(s)
        print(f"  wrote {os.path.relpath(p, ROOT)} (LOTES-2023 ref number)")

    after = appearance_order(io.open(TMPL, encoding="utf-8").read())
    if after != sorted(after):
        raise SystemExit(f"renumbering failed to produce citation order: {after}")
    print(f"  ok    citation numbering is now first-appearance order ({len(after)} refs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check="--check" in sys.argv))
