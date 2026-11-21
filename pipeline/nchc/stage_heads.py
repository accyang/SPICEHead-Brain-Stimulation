"""Stage CHARM head folders for the NCHC cluster under neutral codes (run by the author on the desktop).

Copies only what the mesh and leadfield stages need (label_prep/, segmentation/, surfaces/, eeg_positions/,
T1.nii.gz, settings.ini, final_tissues*, electrodes_1020.json, roi_coords.json), renames the folder to
m2m_h### and scrubs the study ID from text files. The code -> CaseID map stays in the staging folder
on C: and must never leave the lab or enter the code deposit.

  python pipeline/nchc/stage_heads.py SUBJ01                 one head (becomes h001)
  python pipeline/nchc/stage_heads.py --list cohort_8.csv     every head in a CSV with a CaseID column
  python pipeline/nchc/stage_heads.py --all                   all 300 heads (about 63 GB, one tar each)

Output: C:\\spicehead_v22_logs\\nchc_stage\\m2m_h###.tar.gz and head_codes.csv.
Transfer afterwards with sftp/rsync to the cluster's data-transfer node (see README_NCHC.md).
"""
import csv, hashlib, os, shutil, sys, tarfile, time

CH = os.environ.get("SPICEHEAD_CHARM_ROOT", r"L:\BrainProbe\SPICEhead\charm")
COHORT = os.environ.get("SPICEHEAD_COHORT_CSV", r"L:\BrainProbe\SPICEheadv2\SPICEheadv2_cohort.csv")
STAGE = os.environ.get("SPICEHEAD_NCHC_STAGE", r"C:\spicehead_v22_logs\nchc_stage")
KEEP_DIRS = ["label_prep", "segmentation", "surfaces", "eeg_positions"]
KEEP_FILES = ["T1.nii.gz", "settings.ini", "electrodes_1020.json", "roi_coords.json",
              "final_tissues.nii.gz", "final_tissues_LUT.txt", "charm_log.html"]


def m2m_dir(cid):
    return os.path.join(CH, "m2m_SUBJ02asc" if cid.startswith("SUBJ02") else "m2m_" + cid)


def codes():
    p = os.path.join(STAGE, "head_codes.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []
    return p, rows


def stage(cid):
    os.makedirs(STAGE, exist_ok=True)
    p, rows = codes()
    known = {r["CaseID"]: r["code"] for r in rows}
    code = known.get(cid) or f"h{len(rows) + 1:03d}"
    src, dst = m2m_dir(cid), os.path.join(STAGE, "m2m_" + code)
    tar = dst + ".tar.gz"
    if os.path.exists(tar):
        print(f"{cid} -> {code} already staged"); return code
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for d in KEEP_DIRS:
        if os.path.isdir(os.path.join(src, d)):
            shutil.copytree(os.path.join(src, d), os.path.join(dst, d))
    for f in KEEP_FILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    sid = os.path.basename(src)[4:].encode()
    for root, _, files in os.walk(dst):
        for f in files:
            if f.endswith((".ini", ".html", ".txt", ".json")):
                fp = os.path.join(root, f); b = open(fp, "rb").read()
                if sid in b or cid.encode() in b:
                    open(fp, "wb").write(b.replace(sid, code.encode()).replace(cid.encode(), code.encode()))
    with tarfile.open(tar, "w:gz") as t:
        t.add(dst, arcname="m2m_" + code)
    shutil.rmtree(dst)
    sha = hashlib.sha256(open(tar, "rb").read()).hexdigest()
    if cid not in known:
        new = not rows
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["code", "CaseID", "staged", "sha256", "bytes"])
            w.writerow([code, cid, time.strftime("%Y-%m-%d %H:%M"), sha, os.path.getsize(tar)])
    print(f"{cid} -> {code}  {os.path.getsize(tar)/1e6:.0f} MB  sha256 {sha[:16]}")
    return code


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "--all":
        ids = [r["CaseID"] for r in csv.DictReader(open(COHORT, encoding="utf-8-sig"))]
    elif a[0] == "--list":
        ids = [r["CaseID"] for r in csv.DictReader(open(a[1], encoding="utf-8-sig"))]
    else:
        ids = a
    for cid in ids:
        stage(cid)
