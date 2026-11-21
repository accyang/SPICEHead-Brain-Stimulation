# -*- coding: utf-8 -*-
"""Consistency gate across every document.

Third-round review, Major Comment 1. The previous version of this file audited a hand-written list
of expected strings. It passed 85 checks while three sections of the manuscript still carried
pre-recompute numbers — §3.6 aging, the §3.7 focality-sensitivity table and the thalamic intensity —
because nobody had thought to register them. A checklist cannot find the entry that is missing from
the checklist.

So the design changed. The manuscript is now *generated* from `04_manuscript.tmpl.md` plus
`values.py`, and this file's first and most important job is to prove the committed manuscript is
still exactly what the template renders. Drift there is impossible rather than detected: there is
no second place for a number to live.

What remains genuinely checkable:

  1. the manuscript is the template's output           (render_docs --check)
  2. the response letter, which is prose and not templated, still quotes the live numbers
  3. no document asserts a superseded value or a refuted claim
  4. SPICEhead_ANALYSES.md is the current output of its generator
  5. completeness: placeholders, warning markers, length limits
  6b. one canonical montage identifier across every result file
  7. submission-readiness: scaffolding that must not ship (fails under --submission)
  8. the documents agree with *each other* — title, cross-references, headline number,
     bibliography. Generation keeps one document consistent with the data; it does nothing
     for consistency between documents, and the 4,000-word trim broke three of these silently.

Run:  python pipeline/audit_consistency.py     (exit 1 on failure, so it can gate a build)
"""
import json
import os
import re
import subprocess
import sys

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, _here)
from values import values, ROIS, jload, rows_of, R as MONTDIR   # noqa: E402

PY = sys.executable
BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
DOCS = {
    "manuscript": os.path.join(BS, "04_manuscript.md"),
    "letter": os.path.join(BS, "05_response_to_review.md"),
    "checklist": os.path.join(BS, "00_SUBMISSION_CHECKLIST.md"),
    "analyses": os.path.join(ROOT, "Paper_A_SPICEhead", "history_pre_v22", "notes", "SPICEhead_ANALYSES.md"),
    "supplement": os.path.join(BS, "06_supplement.md"),
}

V = values()
fails, warns, oks = [], [], 0

# Regenerate the analyses record BEFORE loading any document, so the pattern checks below see the
# current content rather than a stale copy this run is about to replace. A single run is then
# authoritative — an earlier version needed two passes to settle, which is exactly the kind of
# "passed, but not really" behaviour this file exists to prevent.
_ana = DOCS["analyses"]
_before = open(_ana, encoding="utf-8").read() if os.path.exists(_ana) else None
_gen = subprocess.run([PY, os.path.join(_here, "gen_analyses_md.py")],
                      capture_output=True, text=True, encoding="utf-8", errors="replace",
                      env={**os.environ, "PYTHONIOENCODING": "utf-8"})
_after = open(_ana, encoding="utf-8").read() if os.path.exists(_ana) else None

RAW, T = {}, {}
for name, path in DOCS.items():
    if os.path.exists(path):
        RAW[name] = open(path, encoding="utf-8").read()
        # collapse whitespace (markdown soft-wraps split phrases across lines) and strip emphasis,
        # so bolding a number inside a table cannot break an otherwise correct match
        T[name] = re.sub(r"\s+", " ", RAW[name]).replace("*", "")
    else:
        warns.append((name, "file absent"))


def must_contain(doc, s, label):
    global oks
    if doc not in T:
        return
    if s.replace("*", "") in T[doc]:
        oks += 1
    else:
        fails.append((doc, f"{label}: expected {s!r}"))


def must_not_contain(doc, pattern, why):
    global oks
    if doc not in T:
        return
    if re.search(pattern, T[doc]):
        fails.append((doc, f"/{pattern}/ — {why}"))
    else:
        oks += 1


print("=" * 78)
print("CONSISTENCY AUDIT")
print("=" * 78)
print(f"envelope of record: {V['envelope']}   FEM subjects: {V['fem.1mm.n']}   "
      f"multi-montage resolutions: {V['mm.resolutions']}")
if V["envelope"] != "maxdir":
    fails.append(("source data", f"Table 3 envelope is {V['envelope']!r}, expected 'maxdir'"))

# ---- 1. the manuscript is the template's output --------------------------------
print("\n--- 1. manuscript is generated, not hand-edited ---")
r = subprocess.run([PY, os.path.join(_here, "render_docs.py"), "--check"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
print((r.stdout or "").rstrip())
if r.returncode != 0:
    fails.append(("manuscript", "differs from the template render — run render_docs.py"))
    if r.stderr:
        print("   ", (r.stderr or "").strip().splitlines()[-1])
else:
    oks += 1

# ---- 2. the letter is prose; check it against the live values ------------------
print("\n--- 2. response letter quotes live numbers ---")
for tag, dx in (("1mm", "1 mm"), ("2mm", "2 mm")):
    must_contain("letter", f"{V[f'fem.{tag}.r']:.3f}", f"FEM r @{dx}")
    must_contain("letter", f"{V[f'fem.{tag}.slope']:.3f}", f"FEM slope @{dx}")
    must_contain("letter", f"{V[f'raw.{tag}.prop_bias_slope']:+.3f}", f"proportional bias @{dx}")
    must_contain("letter", f"{V[f'raw.{tag}.ccc']:.3f}", f"CCC @{dx}")
# Both resolutions: the letter's third-round section reports the 1 mm result against the 2 mm one,
# so a stale figure on either side would misstate what refining the grid changed.
for tag in (f"{d:g}mm" for d in V["mm.resolutions"]):
    for t in ("M1", "THAL", "HIPP", "INFERIOR"):
        must_contain("letter", f"{V[f'mm.{tag}.{t}.slope']:.3f}", f"{tag} slope for {t}")
    must_contain("letter", f"{V[f'mm.{tag}.slope_spread']:.3f}", f"{tag} slope spread")
    for k in ("rank_int", "rank_foc"):
        must_contain("letter", f"{V[f'mm.{tag}.{k}.agree']}/{V[f'mm.{tag}.{k}.n']}",
                     f"{tag} {k} agreement")
        must_contain("letter", f"{V[f'mm.{tag}.{k}.rho']:.3f}", f"{tag} {k} Spearman")
    must_contain("letter", f"{V[f'mm.{tag}.fratio_mean']:.3f} ± {V[f'mm.{tag}.fratio_sd']:.3f}",
                 f"{tag} F ratio")
    must_contain("letter", f"{V[f'mm.{tag}.rank_foc.ACC.agree']}/8", f"{tag} ACC rank agreement")
    must_contain("letter", f"{V[f'mm.{tag}.loss.ACC.worst']:.1f}%", f"{tag} ACC worst loss")
    must_contain("letter", f"{V[f'mm.{tag}.loss.insula.worst']:.1f}%", f"{tag} insula worst loss")
must_contain("letter", f"{V['mm.2mm.slope_lo']:.3f} to {V['mm.2mm.slope_hi']:.3f}", "slope range")
must_contain("letter", f"{V['mm.2mm.slope_spread_pct']:.1f}% of", "spread as % of mean")
must_contain("letter", f"{V['mm.2mm.within_sd']:.3f}", "within-montage sd")
must_contain("letter", f"{V['mm.2mm.fratio_lo']:.3f} to {V['mm.2mm.fratio_hi']:.3f}",
             "F ratio range")
must_contain("letter", f"{V['loso.same']} of {V['loso.n']}", "LOSO transfer==oracle")
must_contain("letter", f"{V['loso.mean']:.2f}%", "LOSO mean loss")
must_contain("letter", f"{V['loso.worst']:.1f}%", "LOSO worst loss")
print(f"  {oks} letter values matched" if not fails else f"  {len(fails)} letter mismatches")

# ---- 3. superseded values and refuted claims -----------------------------------
print("\n--- 3. superseded values / refuted claims ---")
# Values: forbidden in the manuscript, where they would read as current claims. The letter is
# allowed to quote them in before/after tables describing what changed.
SUPERSEDED = [
    (r"\+18\.6%", "per-axis M1 steering gain"),
    (r"\+10\.7%", "per-axis insula steering gain"),
    (r"115 ?/ ?300", "per-axis thalamus F>1"),
    (r"125 ?/ ?300", "per-axis insula F>1"),
    (r"Pz\+O2 / F3\+F4", "per-axis M1 montage"),
    (r"Fp1\+Fp2 / P3\+P4", "per-axis insula montage"),
    (r"3\.3% or less", "per-axis ratio max"),
    (r"3\.9% or less", "per-axis 10-10 max"),
    (r"−13\.8%|-13\.8%", "per-axis thalamus aging (third-round finding)"),
    (r"−12\.7%|-12\.7%", "per-axis ACC aging (third-round finding)"),
    (r"−9\.5%|-9\.5%", "per-axis insula aging (third-round finding)"),
    (r"0\.085 V/m", "per-axis thalamic intensity (third-round finding)"),
    (r"r = 0\.78.0\.88.{0,40}(precision|minimal deviation)", "correlation-as-agreement"),
]
for pat, why in SUPERSEDED:
    must_not_contain("manuscript", pat, why)

# Claims: banned everywhere, including in text describing them as retracted — the letter must
# paraphrase when recounting what changed, so no form of the original wording survives.
REFUTED = [
    (r"insensitive to a common scale factor", "refuted by the four-montage slope test"),
    (r"are far less sensitive to a common scale", "refuted by the four-montage slope test"),
    (r"[Aa]nd the Bland.Alt?man bias is ~?0 by construction", "the primary analysis is now raw"),
    # broad on purpose: an earlier narrow pattern missed "reported as upper bounds throughout"
    (r"(treated|reported|described) as upper bounds",
     "not justified: 1 mm has not converged (third round, comment 4)"),
    (r"magnitudes are upper bounds", "not justified: 1 mm has not converged"),
    (r"realistic stimulator front-end", "no transistor-level front end is simulated"),
    (r"never converges", "refuted by the regret analysis: regret ~0 at every n"),
    # fourth round: the electrode-vs-interface inference was withdrawn (the montage dependence
    # does not separate the mechanisms), and the site-invariance phrasing over-generalized
    (r"points the diagnosis at the electrode", "withdrawn inference (fourth round, comment 3)"),
    (r"[Ee]vidence points to the electrode representation",
     "withdrawn inference (fourth round, comment 3)"),
    (r"not fragile to acquisition site", "over-generalizes four sites (fourth round, comment 9)"),
]
for doc in ("manuscript", "letter", "analyses", "supplement"):
    for pat, why in REFUTED:
        must_not_contain(doc, pat, why)
print(f"  {len(SUPERSEDED)} superseded patterns + {len(REFUTED)} refuted claims × 3 documents")

# ---- 4. the analyses record is its generator's current output ------------------
print("\n--- 4. SPICEhead_ANALYSES.md is freshly generated ---")
if _gen.returncode != 0:
    fails.append(("analyses", "generator failed to run"))
    print("   ", (_gen.stderr or "").strip().splitlines()[-1] if _gen.stderr else "")
elif _before != _after:
    warns.append(("analyses", "was stale; regenerated at the start of this run"))
    print("  warn  was stale; regenerated at the start of this run (checks above used the new file)")
else:
    oks += 1
    print("  ok    matches its generator")

# ---- 5. Table 3 montages present --------------------------------------------------
print("\n--- 5. Table 3 optimal montages present in the manuscript ---")
for r in ROIS:
    mon = V[f"t3.{r}.montage"]
    here = "manuscript" in T and mon in T["manuscript"]
    print(f"  {'ok  ' if here else 'warn'}  {r:<15} {mon}")
    if not here:
        warns.append(("manuscript", f"Table 3 montage for {r} absent: {mon}"))

# ---- 6. completeness ---------------------------------------------------------------
print("\n--- 6. draft completeness ---")
for n in ("manuscript", "letter", "checklist", "analyses"):
    if n not in T:
        continue
    todo = RAW[n].count("⚠")
    ph = len(re.findall(r"\*\[Migrate", RAW[n]))
    flag = "warn" if (todo or ph) else "ok  "
    print(f"  {flag}  {n:<12} {todo} warning markers, {ph} placeholders")
    if todo or ph:
        warns.append((n, f"{todo} warning markers, {ph} placeholders"))

# ---- 6b. montage identifiers agree across every result file ----------------------
# Brain Stimulation review, comment 8. Montage keys are frozensets, so their string labels once
# depended on per-process set ordering; a cross-run comparison silently matched only part of its
# candidate list. Labels are canonical now, and this check keeps them that way.
print("\n--- 6b. montage identifiers consistent across result files ---")


def _canon(ms):
    return " / ".join("+".join(p) for p in
                      sorted(tuple(sorted(q.split("+"))) for q in ms.split(" / ")))


try:
    _rt = jload(MONTDIR, "montage_ratio_n300_maxdir.json")["per_roi"]
    _lo = {x["roi"]: x for x in rows_of("v21_generalization", "loso.csv")}
    _d2 = {x["roi"]: x for x in rows_of("v21_dx2_check", "dx2_vs_dx5.csv")}
    _bad = 0
    for r in ROIS:
        ref = _canon(V[f"t3.{r}.montage"])
        for label, got in (("ratio", _rt[r]["montage"] if r in _rt else None),
                           ("dx2 5-mm winner", _d2[r]["winner_5mm"] if r in _d2 else None)):
            if got is None:
                continue
            if _canon(got) != ref:
                fails.append(("results", f"montage label mismatch for {r} ({label}): "
                                         f"{_canon(got)} vs Table 3 {ref}"))
                _bad += 1
            else:
                oks += 1
    print(f"  {'ok  ' if not _bad else 'FAIL'}  Table 3 vs ratio and 2-mm files: "
          f"{_bad} mismatch(es) across {len(ROIS)} targets")
except Exception as ex:
    warns.append(("results", f"montage cross-check skipped: {type(ex).__name__}: {ex}"))
    print(f"  warn  skipped ({type(ex).__name__})")

# ---- 7. submission-readiness gate ------------------------------------------------
# Fifth-round review, comment 14: development scaffolding must not reach a journal. This does not
# fail the ordinary build — the working draft legitimately carries markers — but `--submission`
# turns every hit into a failure, so the final build cannot ship them.
print("\n--- 7. submission-readiness (scaffolding that must not ship) ---")
SCAFFOLD = [
    ("⚠", "reviewer/author to-do markers"),
    ("✅", "verified-number markers"),
    (r"\[Migrate", "migration placeholders"),
    ("Figure plan", "internal figure-planning material"),
    ("Redundancy to collapse", "internal editing table"),
    (r"results/", "repository paths"),
    (r"pipeline/", "script paths"),
    (r"\br20\b", "references to superseded drafts"),
]
SUBMISSION = "--submission" in sys.argv
for doc in ("manuscript", "supplement"):
    if doc not in RAW:
        continue
    for pat, why in SCAFFOLD:
        n = len(re.findall(pat, RAW[doc]))
        if not n:
            oks += 1
            continue
        msg = f"{n}× {why} (/{pat}/)"
        (fails if SUBMISSION else warns).append((doc, f"scaffolding: {msg}"))
        print(f"  {'FAIL' if SUBMISSION else 'warn'}  {doc:<12}{msg}")
if not SUBMISSION:
    print("  (run with --submission to make these failures before building the final package)")

if "manuscript" in RAW:
    # Brain Stimulation Original Research: 4,000-word body, 250-word structured abstract.
    # The body count uses the same rule as pipeline/wordcount.py — prose only, excluding
    # headings, table rows and figure/table callouts — so the two tools cannot disagree.
    from wordcount import body_words                                    # noqa: E402
    s = RAW["manuscript"]
    ab = s.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
    ab = re.sub(r"\*\*(Background|Methods|Results|Conclusions)\.\*\*|[✅⚠️]", "", ab)
    na = len(ab.split())
    nb, _ = body_words(DOCS["manuscript"])
    print(f"  {'ok  ' if na <= 250 else 'warn'}  abstract {na} words (limit 250)")
    print(f"  {'ok  ' if nb <= 4000 else 'warn'}  body     {nb} words (limit 4000)")
    for lab, n_, lim in (("abstract", na, 250), ("body", nb, 4000)):
        if n_ > lim:
            warns.append(("manuscript", f"{lab} {n_} words over the {lim} limit"))

# ---- 8. the documents agree with each other -----------------------------------
# Checks 1-7 prove each document is consistent with the computed values. This asks whether the
# package is consistent with *itself*: the front matter carries the manuscript's title, the
# cross-references resolve, and the headline number is the same everywhere. The trim that cut the
# body to 4,000 words broke three of these silently — orphaned citations, a stale "§4.6 item 9"
# pointer, and a results table still reporting the superseded static values above prose quoting
# the two-carrier ones.
print("\n--- 8. cross-document consistency ---")
FRONT = {
    "cover": os.path.join(BS, "01_cover_letter.md"),
    "title": os.path.join(BS, "02_title_page.md"),
    "highlights": os.path.join(BS, "03_highlights.md"),
    "figplan": os.path.join(BS, "07_figure_plan.md"),
}
X = {k: open(p, encoding="utf-8").read() for k, p in FRONT.items() if os.path.exists(p)}
X.update({k: v for k, v in RAW.items()})


def x_ck(cond, label, detail=""):
    global oks
    if cond:
        oks += 1
        print(f"  ok    {label}")
    else:
        fails.append(("package", f"{label}{' — ' + detail if detail else ''}"))
        print(f"  FAIL  {label}{' — ' + detail if detail else ''}")


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".")


if "manuscript" in X:
    ms = X["manuscript"]

    # (a) every §N.M pointer names a heading that exists
    heads = set(re.findall(r"^#{2,3} (\d+(?:\.\d+)?)", ms, re.M))
    absent = sorted(set(re.findall(r"§(\d+\.\d+)", ms)) - heads)
    x_ck(not absent, "manuscript section cross-references resolve", f"absent: {absent}")

    # (b) supplement pointers name a supplement heading that exists
    if "supplement" in X:
        sheads = set(re.findall(r"^#{2,3} (S\d+(?:\.\d+)?)", X["supplement"], re.M))
        for doc in ("manuscript", "supplement", "letter"):
            if doc not in X:
                continue
            srefs = set(re.findall(r"\b(S\d\.\d)\b", X[doc])) | set(
                re.findall(r"Supplement(?:ary Material)? (S\d)\b", X[doc]))
            bad = sorted(s for s in srefs
                         if s not in sheads and s.split(".")[0] not in sheads)
            x_ck(not bad, f"{doc}: supplement pointers resolve", f"absent: {bad}")

    # (c) "§4.6 item N" citations must resolve. §4.6 is now a signposted prose paragraph
    #     (First, Second, Third) rather than a numbered list, so any surviving "item N" pointer is
    #     stale by construction and the check simply forbids it.
    tail = ms.split("### 4.6", 1)[-1].split("---", 1)[0]
    n_items = len(re.findall(r"^\d+\. \*\*", tail, re.M))
    cited_items = {int(m) for m in re.findall(r"4\.6 item (\d+)", " ".join(X.values()))}
    x_ck(all(1 <= c <= n_items for c in cited_items) if n_items else not cited_items,
         (f"§4.6 'item N' citations within its {n_items} items" if n_items
          else "§4.6 is prose, and no stale 'item N' pointer survives"),
         f"cited: {sorted(cited_items)}")

    # (d) the title, running title and keywords match the title page
    if "title" in X:
        mt = re.search(r"^# (SPICEhead:.*?)\n", ms, re.M)
        tt = re.search(r"\*\*Title:\*\*(.*?)\n\n", X["title"], re.S)
        x_ck(mt and tt and _norm(mt.group(1)) == _norm(tt.group(1)),
             "manuscript title matches the title page")
        mr = re.search(r"\*\*Running title:\*\*(.*)", ms)
        tr = re.search(r"\*\*Running title:\*\*(.*)", X["title"])
        x_ck(mr and tr and _norm(mr.group(1)) == _norm(tr.group(1)), "running title matches")
        kw = lambda s: {x.strip().lower() for x in re.sub(r"\s+", " ", s).split(";") if x.strip()}
        mk = re.search(r"\*\*Keywords:\*\*(.*?)(?:\n\n|---)", ms, re.S)
        tk = re.search(r"\*\*Keywords:\*\*(.*?)(?:\n\n|---)", X["title"], re.S)
        x_ck(mk and tk and kw(mk.group(1)) == kw(tk.group(1)), "keywords match the title page")

    # (e) the headline number is the two-carrier one everywhere it appears
    if "tc.worst.td.com_mm" in V:
        com, stat = f"{V['tc.worst.td.com_mm']:.1f}", f"{V['tc.worst.com_mm']:.1f}"
        for doc in ("manuscript", "cover", "highlights", "letter"):
            if doc in X:
                x_ck(com in X[doc], f"{doc} quotes the two-carrier displacement {com} mm")
        for doc in ("cover", "highlights", "checklist"):
            if doc in X:
                x_ck(stat not in X[doc], f"{doc} does not quote the superseded static {stat} mm")
        x_ck("CoM, static bound" in ms and "(two-carrier)" in ms,
             "the results table labels its two-carrier and static columns")

    # (f) every citation resolves, and no reference is left uncited
    csl_p = os.path.join(BS, "references.csl.json")
    ris_p = os.path.join(BS, "references.ris")
    if os.path.exists(csl_p):
        n_ref = len(json.load(open(csl_p, encoding="utf-8")))
        if os.path.exists(ris_p):
            x_ck(open(ris_p, encoding="utf-8").read().count("TY  -") == n_ref,
                 f"RIS carries all {n_ref} references")
        cited = set()
        for m in re.findall(r"\[([\d,–—\- ]+)\]", ms):
            for part in (x.strip().replace("–", "-").replace("—", "-")
                         for x in m.split(",")):
                a, _, b = part.partition("-")
                if a.isdigit() and b.isdigit():
                    cited |= set(range(int(a), int(b) + 1))
                elif part.isdigit():
                    cited.add(int(part))
        x_ck(not [c for c in cited if c > n_ref], "every citation is in the bibliography",
             f"out of range: {sorted(c for c in cited if c > n_ref)}")
        orphans = sorted(set(range(1, n_ref + 1)) - cited)
        x_ck(not orphans, "no reference is left uncited by the body", f"orphans: {orphans}")

    # (g) every table carries a caption, captions are numbered 1..N in order of appearance, and
    #     every in-text "Table N" names one that exists. Inserting a table shifts the numbering of
    #     everything after it, which is exactly the drift a generated document should not have.
    caps = [int(m.group(1)) for m in re.finditer(r"^\*\*Table (\d+)\.\*\*", ms, re.M)]
    x_ck(caps == list(range(1, len(caps) + 1)),
         f"table captions numbered 1–{len(caps)} in order", f"found: {caps}")
    n_tables = len(re.findall(r"^\|[^\n]*\|\n\|[-: |]+\|", ms, re.M))
    x_ck(len(caps) == n_tables, "every table has a caption",
         f"{n_tables} tables, {len(caps)} captions")
    ptrs = {int(m) for m in re.findall(r"(?<!\*\*)\bTable (\d+)", ms)}
    x_ck(all(p in caps for p in ptrs), "every in-text 'Table N' reference resolves",
         f"dangling: {sorted(p for p in ptrs if p not in caps)}")

    # (g2) Brain Stimulation Guide for Authors, read 15 Aug 2026 from the canonical page.
    #      Highlights: 3-5 bullets, each at most 85 characters including spaces, in a file whose
    #      name contains "highlights". End matter: Acknowledgements and the generative-AI
    #      declaration are each their own section before the reference list, with the AI section
    #      carrying Elsevier's exact required title.
    hl = os.path.join(BS, "03_highlights.md")
    if os.path.exists(hl):
        bullets = [l.strip()[2:] for l in open(hl, encoding="utf-8") if l.startswith("- ")]
        x_ck(3 <= len(bullets) <= 5, f"highlights: {len(bullets)} bullets (3-5 required)")
        over = [b for b in bullets if len(b) > 85]
        x_ck(not over, "highlights: every bullet within 85 characters",
             f"{len(over)} over: {[len(b) for b in over]}")
    AI_TITLE = ("## Declaration of generative AI and AI-assisted technologies in the manuscript "
                "preparation process")
    for head in ("## Acknowledgements", AI_TITLE, "## CRediT authorship contribution statement",
                 "## Declaration of competing interest", "## Funding", "## References"):
        x_ck(head in ms, f"end matter present: {head[3:][:52]}")
    x_ck(ms.find(AI_TITLE) < ms.find("## References"),
         "the generative-AI declaration precedes the reference list")
    x_ck(ms.find("## Acknowledgements") < ms.find("## References"),
         "acknowledgements precede the reference list")

    # (g2b) The submission copies are derived, not maintained. They must be current with the
    #       working copies and must carry none of the working annotations.
    _sub = subprocess.run([PY, os.path.join(_here, "strip_scaffolding.py"), "--check"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    x_ck(_sub.returncode == 0, "submission copies carry no ✅/⚠️ marks or repository paths",
         (_sub.stdout or "").strip().splitlines()[-1] if _sub.stdout else "")
    for nm in ("04_manuscript_submission.md", "06_supplement_submission.md",
               "04_manuscript_zotero.md"):
        pth = os.path.join(BS, nm)
        src = os.path.join(BS, nm.split("_submission")[0].split("_zotero")[0] + ".md")
        x_ck(os.path.exists(pth) and os.path.getmtime(pth) >= os.path.getmtime(src),
             f"{nm} exists and is not older than its source")

    # (g2c) The four bibliography formats must agree, and every DOI must have been resolved.
    _z = subprocess.run([PY, os.path.join(_here, "zotero_export.py")],
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    x_ck(_z.returncode == 0, "CSL-JSON, RIS, BibTeX and Vancouver agree on count and DOI set",
         (_z.stdout or "").strip().splitlines()[-1] if _z.stdout else "")
    _ver = os.path.join(ROOT, "results", "doi_verification.csv")
    if os.path.exists(_ver):
        import csv as _csv
        vr = list(_csv.DictReader(open(_ver, encoding="utf-8")))
        unresolved = [r for r in vr if r["status"] == "UNRESOLVED"]
        mismatch = [r for r in vr if r["status"] == "MISMATCH"]
        x_ck(not unresolved, f"every DOI resolves ({len(vr)} checked)",
             f"unresolved: {[r['n'] for r in unresolved]}")
        x_ck(not mismatch, "stored reference metadata matches the registered record",
             f"mismatched: {[r['n'] for r in mismatch]}")

    # (g2d) The Zotero field-code .docx is hand-written OOXML, so it is checked structurally:
    #       balanced complex fields, valid CSL-JSON payloads carrying full itemData, and a
    #       bibliography and preferences field.
    _zf = subprocess.run([PY, os.path.join(_here, "verify_zotero_docx.py")],
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    x_ck(_zf.returncode == 0, "Zotero field-code .docx is structurally valid",
         (_zf.stdout or "").strip().splitlines()[-1] if _zf.stdout else "")

    # (g2e) every built .docx carries the house formatting of the reference manuscript
    _hf = subprocess.run([PY, os.path.join(_here, "check_house_format.py")],
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    x_ck(_hf.returncode == 0, "every .docx matches the NMH reference formatting",
         (_hf.stdout or "").strip().splitlines()[-1] if _hf.stdout else "")

    # (g2f) the stated word counts on the title block are the measured ones
    mi = re.search(r"\*\*Word Counts:\*\*\s*([\d,]+) words in abstract; ([\d,]+) words in main",
                   ms)
    if mi:
        from wordcount import body_words as _bw
        _nb, _ = _bw(DOCS["manuscript"])
        _ab = ms.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
        _ab = re.sub(r"\*\*(Background|Methods|Results|Conclusions)\.\*\*|[✅⚠️]", "", _ab)
        x_ck(int(mi.group(1).replace(",", "")) == len(_ab.split())
             and int(mi.group(2).replace(",", "")) == _nb,
             "the stated abstract and body word counts are the measured ones",
             f"stated {mi.group(1)}/{mi.group(2)}, measured {len(_ab.split())}/{_nb}")

    # (g3) unified-academic-style-prompt.md: no dash used as sentence punctuation.
    _s = subprocess.run([PY, os.path.join(_here, "style_check.py")],
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    x_ck(_s.returncode == 0, "no dash used as sentence punctuation in manuscript or supplement",
         (_s.stdout or "").strip().splitlines()[-1] if _s.stdout else "")

    # (h) Elsevier numbered style: [1] is the first reference cited, [2] the second, ...
    _r = subprocess.run([PY, os.path.join(_here, "renumber_refs.py"), "--check"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    x_ck(_r.returncode == 0, "citation numbering is in first-appearance order",
         (_r.stdout or "").strip().splitlines()[-1] if _r.stdout else "")

print("\n" + "=" * 78)
print(f"PASSED: {oks}    FAILURES: {len(fails)}    WARNINGS: {len(warns)}")
for d, m in fails:
    print(f"  FAIL  [{d}] {m}")
for d, m in warns:
    print(f"  warn  [{d}] {m}")
print("=" * 78)
sys.exit(1 if fails else 0)
