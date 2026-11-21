# -*- coding: utf-8 -*-
"""Regenerate the per-subject TI-envelope volumes used by the aging analysis.

analyze_rois.py reads precomputed volumes from <m2m>/<envdir>/, so switching the envelope
definition means regenerating them. Writes to a NAMED subdirectory (default envelopes_maxdir) so
the published per-axis volumes in <m2m>/envelopes/ are left intact.

Resumable and shardable: a subject whose summary.json already exists is skipped.

  python pipeline/regen_envelopes_maxdir.py <cohort.csv> <charm_outroot> [envdir] [shard] [nshards]
"""
import csv
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
sys.path.insert(0, ROOT)
sys.path.insert(0, _here)

import run_cohort_envelopes as RCE


def main(cohort_csv, outroot, envdir="envelopes_maxdir", shard=0, nshards=1):
    mode = os.environ.get("SPICEHEAD_ENVELOPE", "maxdir")
    ids = [r["CaseID"] for r in csv.DictReader(open(cohort_csv, encoding="utf-8-sig"))]
    if nshards > 1:
        ids = [c for i, c in enumerate(ids) if i % nshards == shard]
    print(f"shard {shard}/{nshards}: {len(ids)} subjects, envelope={mode}, out=<m2m>/{envdir}",
          flush=True)

    done = skipped = failed = 0
    t0 = time.time()
    for i, cid in enumerate(ids, 1):
        m2m = os.path.join(outroot, "m2m_SUBJ01asc" if cid.startswith("SUBJ01") else "m2m_" + cid)
        # eeg_electrodes.json holds exactly the 8 sites of the two aging montages
        # (FC3-C1/C3-CP3 and F7-PO7/F8-PO8). C1 is absent from the 32-position 10-10 file and
        # PO7/PO8/FC3/CP3 from the 10-20 files, so this is the only complete source.
        ejson = os.path.join(m2m, "eeg_electrodes.json")
        out = os.path.join(m2m, envdir)
        if os.path.exists(os.path.join(out, "summary.json")):
            done += 1
            print(f"  [{i}/{len(ids)}] {cid} (cached)", flush=True)
            continue
        if not os.path.exists(ejson):
            skipped += 1
            print(f"  [{i}/{len(ids)}] {cid} - no electrode file, skip", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        t = time.time()
        try:
            RCE.main(m2m, ejson, out)
            done += 1
            print(f"  [{i}/{len(ids)}] {cid} ({time.time()-t:.0f}s)", flush=True)
        except SystemExit as e:
            failed += 1
            print(f"  [{i}/{len(ids)}] {cid} FAILED: {e}", flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(ids)}] {cid} FAILED {type(e).__name__}: {e}", flush=True)
    print(f"\nshard {shard}: {done} done, {skipped} skipped, {failed} failed "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "envelopes_maxdir",
         int(sys.argv[4]) if len(sys.argv) > 4 else 0,
         int(sys.argv[5]) if len(sys.argv) > 5 else 1)
