# -*- coding: utf-8 -*-
"""Bring the Forerunner 1 part of the Paper A v23 recompute into results_v23 under case IDs (28 Oct 2026).

The cluster saw h-codes only. This copies C:/spicehead_v23_logs/f1_fetch/results_v23 (fetched and SHA-256 verified)
into the project's results_v23, replacing every h-code (a whole token h###) by its case ID from the private code map:
in file names, and inside JSON/CSV/TXT files. Pickles carry no codes (only their names do) and are copied byte for byte.
Afterwards no h-code may remain in any copied file; the outputs then match what a desktop run of the same jobs writes.
    python pipeline/nchc/f1_relabel_v23.py [--dry-run]
"""
import csv
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = r"C:\spicehead_v23_logs\f1_fetch\results_v23"
DST = os.path.join(ROOT, "results_v23")
CODEMAP = r"C:\spicehead_v22_logs\nchc_stage\head_codes.csv"                     # private; read only
TOKEN = re.compile(r"(?<![A-Za-z0-9])h\d{3}(?![0-9])")      # h### not inside a longer word or number; "_" may follow
TEXT = (".json", ".csv", ".txt", ".md")


def main():
    global SRC
    dry = "--dry-run" in sys.argv
    if "--src" in sys.argv:                                                    # a later fetch (e.g. f1_fetch2/results_v23)
        SRC = sys.argv[sys.argv.index("--src") + 1]
    cmap = {r["code"]: r["CaseID"] for r in csv.DictReader(open(CODEMAP, encoding="utf-8-sig"))}
    sub = lambda s: TOKEN.sub(lambda m: cmap[m.group(0)], s)
    n_files = n_tokens = 0
    for dirpath, _, files in os.walk(SRC):
        for f in files:
            src = os.path.join(dirpath, f)
            rel = os.path.relpath(src, SRC)
            dst = os.path.join(DST, os.path.dirname(rel), sub(os.path.basename(rel)))
            n_files += 1
            if dry:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if f.lower().endswith(TEXT):
                t = open(src, encoding="utf-8").read()
                n_tokens += len(TOKEN.findall(t))
                open(dst, "w", encoding="utf-8", newline="").write(sub(t))
            else:
                shutil.copyfile(src, dst)
    left = []
    if not dry:                                                                # no h-code may remain
        for dirpath, _, files in os.walk(SRC):
            for f in files:
                rel = os.path.relpath(os.path.join(dirpath, f), SRC)
                dst = os.path.join(DST, os.path.dirname(rel), sub(os.path.basename(rel)))
                if TOKEN.search(os.path.basename(dst)) or (f.lower().endswith(TEXT) and TOKEN.search(open(dst, encoding="utf-8").read())):
                    left.append(dst)
    print(f"{'would copy' if dry else 'copied'} {n_files} files into {DST}; {n_tokens} h-codes replaced inside text files; files still holding an h-code: {len(left)}")
    if left:
        raise SystemExit("\n".join(left[:10]))


if __name__ == "__main__":
    main()
