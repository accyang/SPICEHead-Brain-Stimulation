#!/usr/bin/env bash
# Forerunner 1: check and unpack the 300 staged head tarballs (Paper A v23 recompute, 26 Oct 2026). Each tarball is
# checked against jobs/code_sha.txt before it is unpacked into data/; data/m2m_<code>/.ok marks a verified head.
#   sbatch ~/f1_unpack.sbatch   (run from the driver)
#SBATCH --job-name=f1_unpack
#SBATCH --account=<project>
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=112
#SBATCH --time=02:00:00
#SBATCH --output=/work1/<account>/spicehead/logs/f1_unpack_%j.log
set -uo pipefail
P=/work1/<account>/spicehead
export RD=${RUNDIR:-$P}                 # run folder (data, results_v23, logs); tarballs and lists stay in $P
mkdir -p $RD/data $RD/logs
one() {
  local c=$1 P=/work1/<account>/spicehead
  [ -e $RD/data/m2m_$c/.ok ] && return 0
  local want got; want=$(awk -v c=$c '$1==c{print $2}' $P/jobs/code_sha.txt); got=$(sha256sum $P/tar/m2m_$c.tar.gz | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$c checksum mismatch"; return 1; }
  rm -rf $RD/data/m2m_$c && tar -xzf $P/tar/m2m_$c.tar.gz -C $RD/data && touch $RD/data/m2m_$c/.ok && echo "$c ok"
}
export -f one
ls $P/tar/ | sed -n 's/^m2m_\(h[0-9]*\)\.tar\.gz$/\1/p' | xargs -P 32 -I{} bash -c 'one {}'
n=$(ls $RD/data/m2m_h*/.ok 2>/dev/null | wc -l)
echo "unpacked and verified: $n of 300"
[ "$n" = 300 ]                          # a non-zero exit holds the dependent stage-A job (afterok)
