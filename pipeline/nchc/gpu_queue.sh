#!/bin/bash
# run_on_gpus <function> <ngpu> <item>...  (2 Nov 2026)
# Runs "<function> <item> <gpu index>" with one worker per GPU: a GPU takes the next item as soon as it is free.
# This replaces the loop that launched "one $c $((i % NGPU))" when any slot opened. That loop chose the GPU by the
# item's position, not by which GPU had just become free, so in the 300-head uncertainty run (array 660766) two heads
# shared a GPU for 181 GPU-hours while other GPUs sat idle. Source this file from a job script; needs flock.
run_on_gpus() {
  local fn=$1 ngpu=$2 q g; shift 2
  q=$(mktemp "${TMPDIR:-/tmp}/gpu_queue.XXXXXX") || return 1
  printf '%s\n' "$@" > "$q"
  _gpu_worker() {
    local gpu=$1 item
    while :; do
      item=$(flock "$q.lock" bash -c 'read -r x < "$0" || true; sed -i 1d "$0"; printf "%s" "$x"' "$q")
      [ -n "$item" ] || break
      "$fn" "$item" "$gpu"
    done
  }
  for g in $(seq 0 $((ngpu - 1))); do _gpu_worker "$g" & done
  wait
  rm -f "$q" "$q.lock"
}
