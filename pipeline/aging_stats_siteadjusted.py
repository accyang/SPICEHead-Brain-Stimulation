"""
Site-adjusted cross-sectional age statistics for the SPICEhead cohort.

Addresses reviewer comment H (SPICEhead_Critical_Review): the four acquisition
sites differ markedly in age (Vet_Home ~79 y vs Bai_TVGH ~38 y), so unadjusted
age slopes conflate age with site/scanner. This script reports, for every ROI
and both montages:

  - unadjusted OLS:            env ~ Age
  - site+sex-adjusted OLS:     env ~ Age + C(Site) + C(Sex)

with 95% CIs, BH-FDR q-values across the six ROIs, and the fitted percent
change from age 25 to 85 (relative to the cohort-mean envelope). Both the
brain-ROI and grey-matter-ROI definitions are processed.

Inputs (synced via Dropbox):
  results/aging_n300_brain/cohort_roi_metrics.csv
  results/aging_n300_gm/cohort_roi_metrics.csv
Output:
  results/aging_siteadjusted/aging_stats.csv
  results/aging_siteadjusted/site_age.csv

Run:  python pipeline/aging_stats_siteadjusted.py
"""

from __future__ import annotations

import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

TARGETS = ["M1_handknob", "thalamus", "insula", "ACC",
           "hippocampus_L", "hippocampus_R"]
MONTAGES = ["thalamic", "motor"]
ROI_DEFS = {
    "brain": "results/aging_n300_brain/cohort_roi_metrics.csv",
    "gm": "results/aging_n300_gm/cohort_roi_metrics.csv",
}
OUT = "results/aging_siteadjusted"


def fit_target(df, montage, target):
    y = f"{montage}_{target}_mean"
    d = df[["Age", "Sex", "Site", y]].dropna().rename(columns={y: "env"})
    base = d["env"].mean()
    mu = smf.ols("env ~ Age", d).fit()
    ma = smf.ols("env ~ Age + C(Site) + C(Sex)", d).fit()
    ci = ma.conf_int().loc["Age"]
    return {
        "montage": montage, "target": target, "n": int(d.shape[0]),
        "slope_unadj": mu.params["Age"], "p_unadj": mu.pvalues["Age"],
        "pct_unadj": 100 * mu.params["Age"] * 60 / base,
        "slope_adj": ma.params["Age"], "p_adj": ma.pvalues["Age"],
        "ci_lo": ci[0], "ci_hi": ci[1],
        "pct_adj": 100 * ma.params["Age"] * 60 / base,
    }


def main():
    os.makedirs(OUT, exist_ok=True)

    # Site-age confound table (from the GM file; demographics are identical).
    gm = pd.read_csv(ROI_DEFS["gm"], encoding="latin-1")
    site_age = gm.groupby("Site")["Age"].agg(["count", "mean", "min", "max"]).round(1)
    site_age.to_csv(os.path.join(OUT, "site_age.csv"))
    print("Age by site (age-site confound):")
    print(site_age.to_string(), "\n")

    rows = []
    for roi, path in ROI_DEFS.items():
        df = pd.read_csv(path, encoding="latin-1")
        for montage in MONTAGES:
            block = [fit_target(df, montage, t) for t in TARGETS]
            q = multipletests([b["p_adj"] for b in block], method="fdr_bh")[1]
            for b, qi in zip(block, q):
                b["roi_def"] = roi
                b["q_adj_fdr"] = qi
                rows.append(b)

    res = pd.DataFrame(rows)[[
        "roi_def", "montage", "target", "n",
        "slope_unadj", "pct_unadj", "p_unadj",
        "slope_adj", "ci_lo", "ci_hi", "pct_adj", "p_adj", "q_adj_fdr",
    ]]
    out_csv = os.path.join(OUT, "aging_stats.csv")
    res.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({res.shape[0]} rows)")

    # Headline: thalamic montage, both ROI defs.
    for roi in ("brain", "gm"):
        sub = res[(res.roi_def == roi) & (res.montage == "thalamic")]
        print(f"\nThalamic montage — {roi} ROI (site+sex adjusted, FDR):")
        for _, x in sub.iterrows():
            print(f"  {x.target:16s} {x.pct_adj:+6.1f}%  "
                  f"[{x.ci_lo:+.1e},{x.ci_hi:+.1e}]  q={x.q_adj_fdr:.1e}")


if __name__ == "__main__":
    main()
