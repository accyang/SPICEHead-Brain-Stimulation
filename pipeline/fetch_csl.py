# -*- coding: utf-8 -*-
"""Fetch CSL-JSON for every cited DOI by content negotiation, and write the citation-to-DOI map.

DOI content negotiation returns the record in exactly the shape Zotero stores internally:

    GET https://doi.org/{DOI}   Accept: application/vnd.citationstyles.csl+json

That is what makes the next step possible. A Zotero field code carries a complete `itemData`
object per cited work, so the citation can be re-rendered without the item being in the user's
library, and this is the authoritative source for those objects.

Writes:
  results/csl_by_doi.json               DOI -> CSL-JSON item, as returned by doi.org
  Paper_A_SPICEhead/Brain Stimulation/citation_doi_map.md the numbered citation list with its DOI and short form

  python pipeline/fetch_csl.py [--refresh]
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
CACHE = os.path.join(ROOT, "results", "csl_by_doi.json")
ACCEPT = "application/vnd.citationstyles.csl+json"
UA = "SPICEhead-reference-tooling/1.0 (mailto:<email>)"


def negotiate(doi, tries=3):
    """The CSL-JSON record for a DOI, straight from the registration agency."""
    url = "https://doi.org/" + doi
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": ACCEPT, "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8")), None
        except urllib.error.HTTPError as e:
            if e.code in (404, 406):
                return None, f"HTTP {e.code}"
            if k == tries - 1:
                return None, f"HTTP {e.code}"
        except Exception as e:                                        # noqa: BLE001
            if k == tries - 1:
                return None, type(e).__name__
        time.sleep(1.5 * (k + 1))
    return None, "unreachable"


def short(e):
    """Author-year short form for the map."""
    a = e.get("author") or []
    fam = (a[0].get("family") if a else "") or e.get("publisher") or "Anon"
    if len(a) == 2:
        who = f"{fam} & {a[1].get('family', '')}"
    elif len(a) > 2:
        who = f"{fam} et al."
    else:
        who = fam
    try:
        yr = e["issued"]["date-parts"][0][0]
    except (KeyError, IndexError, TypeError):
        yr = "n.d."
    return f"{who}, {yr}"


def main(refresh=False):
    csl = json.load(open(os.path.join(BS, "references.csl.json"), encoding="utf-8"))
    csl.sort(key=lambda e: e["citation-number"])
    cache = {}
    if os.path.exists(CACHE) and not refresh:
        cache = json.load(open(CACHE, encoding="utf-8"))

    rows, failed = [], []
    for e in csl:
        n, doi = e["citation-number"], e.get("DOI")
        if not doi:
            rows.append((n, "", short(e), e.get("title", ""), "no DOI (software documentation)"))
            print(f"  [{n:2d}] (no DOI)  {short(e)}")
            continue
        if doi in cache:
            item, err = cache[doi], None
        else:
            item, err = negotiate(doi)
            if item:
                cache[doi] = item
            time.sleep(0.2)
        if item is None:
            failed.append((n, doi, err))
            rows.append((n, doi, short(e), e.get("title", ""), f"FAILED: {err}"))
            print(f"  [{n:2d}] FAILED    {doi}  ({err})")
        else:
            rows.append((n, doi, short(item), item.get("title", ""), "ok"))
            print(f"  [{n:2d}] ok        {doi}  {short(item)}")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    out = os.path.join(BS, "citation_doi_map.md")
    L = ["# Citation-to-DOI map", "",
         "Every in-text citation number, the DOI it resolves to, and the short form used in the",
         "Zotero marker copy. CSL-JSON for each DOI was obtained by content negotiation against",
         "`https://doi.org/{DOI}` with `Accept: application/vnd.citationstyles.csl+json`, which is",
         "the record format Zotero itself stores, and is cached in `results/csl_by_doi.json`.", "",
         f"{len(rows)} references, {sum(1 for r in rows if r[1])} with a DOI, "
         f"{len(failed)} unresolved.", "",
         "| # | Short form | DOI | Title | Status |", "|---|---|---|---|---|"]
    for n, doi, sh, ti, st in rows:
        link = f"[{doi}](https://doi.org/{doi})" if doi else "—"
        L.append(f"| {n} | {sh} | {link} | {ti[:70]} | {st} |")
    io.open(out, "w", encoding="utf-8", newline="").write("\n".join(L) + "\n")
    print(f"\n  wrote {os.path.relpath(out, ROOT)} and {os.path.relpath(CACHE, ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(refresh="--refresh" in sys.argv))
