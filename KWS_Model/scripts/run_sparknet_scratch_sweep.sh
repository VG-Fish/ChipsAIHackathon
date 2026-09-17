#!/usr/bin/env bash
#
# Train the from-scratch narrow SparkNet arm: C12, C10 and C8 x seeds 0-4.
#
# WHAT THIS ESTABLISHES
#
# Every narrow SparkNet in this study has so far been *pruned* from the C16
# paper replication and then fine-tuned.  That makes the conventional arm a
# compression result, not an architecture result: when a dendritic candidate
# beats a pruned C12, part of what it beat is the damage pruning did, not the
# capacity a 12-channel network actually has.  A narrow network trained from
# scratch under the same recipe, at matched parameters, is the honest floor --
# it is what a practitioner who simply wanted a smaller model would have
# built.  These runs are that floor.
#
# They are also the starting point of the dendritic arm: the dendritic
# experiment grows dendrites from a base network, and the base it should grow
# from is the best network available at that width, not a pruned one.  So this
# sweep blocks everything downstream.  Nothing measured against a pruned
# baseline is final until these 15 runs exist.
#
# THE RECIPE
#
# configs/train/sparknet_c16_paper.yaml is the paper recipe and is shared by
# every width here, unchanged: SGD with momentum 0.9, the composite objective
# 100*CE + gate_sparsity, a warmup-hold-polynomial learning-rate schedule, and
# 200 epochs.  Holding the recipe fixed across widths is the whole point -- a
# width that got its own tuned schedule would not be comparable to the others
# or to the C16 replication.
#
# COST
#
# 15 runs x 200 epochs.  The models are tiny, but this is not a quick script;
# it runs sequentially by default so it can be left alone on one machine.
#
# USAGE
#
#   bash scripts/run_sparknet_scratch_sweep.sh
#
# Re-running is safe and cheap: a combination whose output directory already
# holds a completed run is skipped, so a sweep interrupted after nine runs
# resumes by being re-invoked.  A single combination can be re-driven without
# editing this file:
#
#   WIDTHS=8 SEEDS=3 bash scripts/run_sparknet_scratch_sweep.sh
#
# Exits non-zero if any run failed, after attempting all of them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WIDTHS="${WIDTHS:-12 10 8}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STAGE="${STAGE:-paper_replication}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparknet-scratch}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/speech_commands_v2_mfcc32_paper.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/sparknet_c16_paper.yaml}"

mkdir -p "$OUTPUT_ROOT"

# Completion markers.  kws.train writes models/checkpoints/<stage>/latest.pt at
# every epoch boundary for resume, but it only materialises best.pt *after* the
# training loop returns -- src/kws/train.py rebuilds it from latest.pt's
# best_model_state_dict once the loop is done -- and then immediately writes
# metrics/summaries.yaml.  So latest.pt says "this run started", while best.pt
# says "this run's training loop finished".  Requiring the summary as well
# closes the one-write window between them: a process killed exactly there
# would otherwise be skipped forever while leaving no phase summary for the
# downstream tooling that reads one.
run_is_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/models/checkpoints/${STAGE}/best.pt" ]] &&
    [[ -f "${output_dir}/metrics/summaries.yaml" ]]
}

statuses=()
failures=0

for width in $WIDTHS; do
  for seed in $SEEDS; do
    output_dir="${OUTPUT_ROOT}/c${width}-seed${seed}"
    log_file="${OUTPUT_ROOT}/c${width}-seed${seed}.log"

    if run_is_complete "$output_dir"; then
      echo "skip  C${width} seed${seed}  (${output_dir} already complete)"
      statuses+=("${width}|${seed}|skipped")
      continue
    fi

    echo "run   C${width} seed${seed}  -> ${output_dir}  (log: ${log_file})"
    # Each run's stdout and stderr are teed to its own file so a failure 140
    # epochs in is diagnosable after the fact rather than only from whatever
    # is still in the terminal scrollback.
    if uv run --env-file .env python -m kws.train \
      --data-config "$DATA_CONFIG" \
      --model-config "configs/model/sparknet_c${width}_paper.yaml" \
      --train-config "$TRAIN_CONFIG" \
      --stage "$STAGE" --seed "$seed" \
      --output-dir "$output_dir" 2>&1 | tee "$log_file"; then
      if run_is_complete "$output_dir"; then
        statuses+=("${width}|${seed}|ok")
      else
        # The command exited 0 without leaving the markers a finished run
        # leaves.  Treat that as a failure rather than letting the next
        # invocation silently re-run it as if nothing had happened.
        echo "WARN  C${width} seed${seed} exited 0 but left no completion marker" >&2
        statuses+=("${width}|${seed}|incomplete")
        failures=$((failures + 1))
      fi
    else
      # One bad seed must not cost the other fourteen runs, so record it and
      # keep going; the exit status at the end is what reports the failure.
      echo "FAIL  C${width} seed${seed}  (see ${log_file})" >&2
      statuses+=("${width}|${seed}|failed")
      failures=$((failures + 1))
    fi
  done
done

echo
echo "sweep summary (${OUTPUT_ROOT})"
printf '%-8s %-6s %s\n' "width" "seed" "status"
printf '%-8s %-6s %s\n' "-----" "----" "------"
for entry in "${statuses[@]}"; do
  IFS='|' read -r width seed status <<<"$entry"
  printf '%-8s %-6s %s\n' "C${width}" "${seed}" "${status}"
done

if ((failures > 0)); then
  echo
  echo "${failures} run(s) did not complete" >&2
  exit 1
fi
