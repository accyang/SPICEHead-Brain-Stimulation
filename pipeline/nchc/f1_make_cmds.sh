#!/usr/bin/env bash
# Forerunner 1: command lists of the Paper A v23 recompute (26 Oct 2026), the cluster part of run_v23_all.py:
#   stageA.txt  2 mm full search in the 20 pinned heads (20 shards), the network census, the 5 mm cohort search and
#               the envelope volumes for the age analysis (150 shards each, two heads per shard)
#   stageB.txt  cohort aggregation, then ratio and sensitivity (both need the cohort optima)
# Paths and arguments follow run_v22_all.py; the site- and age-dependent steps (LOSO, learning curve, 2 mm collection,
# age analysis) run on the desktop after the fetch.
#   bash f1_make_cmds.sh
P=/work1/<account>/spicehead; RD=${RUNDIR:-$P}; L=$RD/logs; R=$RD/results_v23
CSV=$P/jobs/heads300.csv; CH=$RD/data; OPT=$R/montage_optimization; T3DIR=$OPT/cohort_gm300_dx5_maxdir; T3=$T3DIR/cohort_montage_optima.json
run() { echo "cd $P/code && python $2 > $L/$1.log 2>&1 || echo $1 >> $L/FAILED"; }
mkdir -p $L $R
{
  for k in $(seq 0 19); do run t7_s$k "pipeline/v21_dx2_cohort_check.py --score-full 5 --shard $k/20"; done
  run scale_census "pipeline/v22_scale_census.py 5"
  for k in $(seq 0 149); do
    run cohort_$k "pipeline/run_montage_cohort.py $CSV $CH $T3DIR 5.0 0 gm $k 150"
    run aging_$k "pipeline/regen_envelopes_maxdir.py $CSV $CH envelopes_maxdir_v23 $k 150"
  done
} > $RD/stageA.txt
{
  echo "cd $P/code && python pipeline/run_montage_cohort.py $CSV $CH $T3DIR 5.0 0 gm > $L/cohort_agg.log 2>&1 && { python pipeline/montage_ratio.py $CSV $CH $T3 $OPT/montage_ratio_n300_maxdir.json 5.0 > $L/ratio.log 2>&1 || echo ratio >> $L/FAILED; } || echo cohort_agg >> $L/FAILED"
  echo "sleep 5; while [ ! -s $T3 ]; do grep -qx cohort_agg $L/FAILED 2>/dev/null && exit 1; sleep 30; done; cd $P/code && python pipeline/v21_focality_sensitivity.py 5.0 > $L/sensitivity.log 2>&1 || echo sensitivity >> $L/FAILED"
} > $RD/stageB.txt
echo "stage A: $(wc -l < $RD/stageA.txt) commands; stage B: $(wc -l < $RD/stageB.txt) (in $RD)"
