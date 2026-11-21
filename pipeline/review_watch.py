# -*- coding: utf-8 -*-
"""Detect review documents that land in the paper folders (the 10-minute review loop).

  python pipeline/review_watch.py              list new or changed documents (JSON)
  python pipeline/review_watch.py --ack P ...  mark those paths as handled
  python pipeline/review_watch.py --baseline   mark every current document as handled

Watched: .md .docx .pdf .txt files directly in the project root, "Paper_A_SPICEhead/Brain Stimulation/" and
"Paper_A_SPICEhead/Brain Stimulation/v22/". A file is reported as
  "reviews"  if its name suggests a review (review, critique, comment, referee, decision,
             feedback, final) and the pipeline did not generate it;
  "other"    if it is any other new document the pipeline did not generate (a review often
             arrives with a companion file, as the 1 October review did).
A handled file that changes on disk is reported again. State lives outside Dropbox, in
C:\\spicehead_v22_logs\\review_watch.json.
"""
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = os.path.join(ROOT, "Paper_A_SPICEhead", "Brain Stimulation")
FOLDERS = [ROOT, BS, os.path.join(BS, "v22"), os.path.join(BS, "review")]   # reviews land in review/
STATE = r"C:\spicehead_v22_logs\review_watch.json"
EXT = (".md", ".docx", ".pdf", ".txt")
REVIEW = re.compile(r"review|critique|comment|referee|decision|feedback|final", re.I)
# documents this project's own tooling writes; never treated as incoming reviews
GENERATED = re.compile(
    r"^(04_manuscript|06_supplement|RESULTS_v22|RESULTS_final_|RESULTS_verification_|REPLY_|"
    r"references_|ACCEPTANCE_CRITERIA|SPICEhead_v2\d|SPICEhead_ANALYSES|REVISION_LOG|"
    r"PLAN_|BENCH_PROTOCOL|"
    r"00_SUBMISSION|01_cover|02_title|ZOTERO_README|citation_doi_map|\.zotero_build|~\$)", re.I)


def scan():
    out = {}
    for d in FOLDERS:
        if not os.path.isdir(d):
            continue
        # review/ holds only incoming documents; its names (RESULTS_v22_REVIEW_...) would
        # otherwise match the GENERATED prefix and be skipped, as happened on 2 October
        incoming = os.path.basename(d) == "review"
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p) and f.lower().endswith(EXT) and not f.endswith(".tmpl.md") \
                    and (incoming or not GENERATED.match(f)):
                out[os.path.relpath(p, ROOT)] = os.path.getmtime(p)
    return out


def load():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {}


def save(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE + ".part", "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(STATE + ".part", STATE)


if __name__ == "__main__":
    a = sys.argv[1:]
    state, now = load(), scan()
    if "--baseline" in a:
        state.update(now)
        save(state)
        print(f"baseline: {len(now)} documents marked as handled")
    elif "--ack" in a:
        for p in a[a.index("--ack") + 1:]:
            rel = os.path.relpath(os.path.join(ROOT, p), ROOT) if not os.path.isabs(p) \
                else os.path.relpath(p, ROOT)
            if rel in now:
                state[rel] = now[rel]
        save(state)
        print("acknowledged:", ", ".join(a[a.index("--ack") + 1:]))
    else:
        new = sorted(p for p, m in now.items() if state.get(p) != m)
        res = {"checked": time.strftime("%Y-%m-%d %H:%M"),
               "reviews": [p for p in new if REVIEW.search(os.path.basename(p))],
               "other": [p for p in new if not REVIEW.search(os.path.basename(p))]}
        print(json.dumps(res, ensure_ascii=False, indent=1))
