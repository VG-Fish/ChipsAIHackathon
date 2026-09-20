#!/usr/bin/env bash
#
# Grow one PerforatedAI dendrite *during* paper-recipe SparkNet training,
# across widths x seeds x placements.
#
# WHAT THIS ESTABLISHES
#
# The v2 study (scripts/run_sparknet_dendritic_study.py) added dendrites to
# converged scratch networks through a fine-tune and a PAI phase on a different
# optimiser, and every placement moved together -- it measured the optimiser
# handoff as much as the dendrites.  This sweep asks the cleaner question.  Each
# run trains SparkNet from scratch on the paper recipe, grows ONE dendrite at a
# fixed epoch partway through (the switch epoch, 120 in the train config),
# trains only the dendrite's candidate for a short window (15 epochs) with the
# base frozen, and then carries on with the rest of the same schedule.  The
# driver is kws.optimize.sparknet_grow_dendrites; the recipe and its rationale
# are in configs/train/sparknet_grow_dendrites_paper.yaml.
#
# The base optimiser schedule -- SGD with momentum 0.9, 100*CE + gate_sparsity,
# warmup-hold-polynomial learning rate, 200 epochs -- is identical to the one
# the from-scratch runs used.  So the no-dendrite controls already exist and are
# paired with these runs by width and seed:
#
#   C2-C12  outputs/sparknet-dendritic-study-v2/scratch/c{w}-seed{s}
#   C16     outputs/sparknet-paper-replication/c16-seed{s}
#
# Both hold seeds 0-4, and so does the default grid here.  No new control
# training is needed; the difference between a grow run and its control is the
# dendrite and the candidate epochs it cost, not a different recipe.  One
# caveat for C16: its controls were trained with
# configs/train/sparknet_c16_paper.yaml (num_workers 0, uncached features), not
# the fast-io loader the narrow controls and this recipe use.
# The optimiser schedule is the same, but the augmentation RNG stream is not,
# so C16 pairs by recipe and seed rather than bit-for-bit up to the switch.
#
# COST
#
# Each run is ~200 base epochs plus ~15 candidate epochs, roughly 12-13 min on
# this machine.  The default grid is 7 widths x 5 seeds x 2 placements = 70
# runs, about 15 h sequentially.  Runs are sequential so the script can be left
# alone on one machine.
#
# ORDER
#
# Seed is the outermost loop, then width, then placement.  Seed 0 therefore
# finishes at every width and both placements (14 runs, ~3 h) before any
# second seed starts: a sweep that is stopped early still gives a full,
# single-seed picture across the whole width range instead of five seeds of
# one width.  The default width order is deliberately not sorted -- it starts at
# C8 and alternates across the range -- so a sweep stopped partway through a
# seed has still covered both wide and narrow networks.
#
# USAGE
#
# Smoke-test one combination first (~13 min) and check its log and
# reports/grow_summary.yaml before committing the machine for 15 h:
#
#   WIDTHS=8 SEEDS=0 PLACEMENTS=fc bash scripts/run_sparknet_grow_sweep.sh
#
# Then the full sweep:
#
#   bash scripts/run_sparknet_grow_sweep.sh
#
# ARMS
#
# An arm is a placement plus a recipe variant: <placement><ARM_SUFFIX>.  The
# default arm of a placement is the placement itself (ARM_SUFFIX empty) and
# runs the recipe exactly as configured.  A variant passes extra driver flags
# through EXTRA_ARGS and must name its own arm with ARM_SUFFIX, e.g.
#
#   WIDTHS=8 SEEDS=0 PLACEMENTS=fc ARM_SUFFIX=-wd1e-3 EXTRA_ARGS="--dendrite-weight-decay 0.001" \
#     bash scripts/run_sparknet_grow_sweep.sh
#   WIDTHS=8 SEEDS=0 PLACEMENTS=fc ARM_SUFFIX=-switch170 EXTRA_ARGS="--switch-epoch 170" \
#     bash scripts/run_sparknet_grow_sweep.sh
#   WIDTHS=8 SEEDS=0 PLACEMENTS=fc ARM_SUFFIX=-sham EXTRA_ARGS="--sham" \
#     bash scripts/run_sparknet_grow_sweep.sh
#
# Every run gets --arm <arm> (the driver records it as `arm` in its summary),
# then EXTRA_ARGS.  Two guards keep arms apart:
#
# - The sweep refuses to start when EXTRA_ARGS, SWITCH_EPOCH or
#   CANDIDATE_EPOCHS is set but ARM_SUFFIX is empty, even in a dry run: the
#   variant would otherwise land in the default arm's directories, or be
#   skipped because the default arm already completed them.
# - A complete run is skipped only if its summary's `arm` is the arm being
#   run (a summary without `arm`, from before the driver recorded it, counts
#   as its `placement`).  Otherwise the combination is refused: never skipped,
#   never moved aside, never overwritten.
#
# DRY_RUN=1 prints the plan (the skip / move-aside / run decision and the exact
# command for every combination) without running or moving anything.
#
# Environment overrides (defaults in brackets):
#
#   WIDTHS            ["8 12 4 10 6 2 16"]  integer widths or model tokens such
#                     as c4g16; each needs configs/model/sparknet_<token>_paper.yaml
#   SEEDS             ["0 1 2 3 4"]
#   PLACEMENTS        ["fc pointwise"]      keys from grow_dendrites.placements in TRAIN_CONFIG
#   ARM_SUFFIX        [empty]  appended to the placement to name the arm;
#                     letters, digits, '.', '_' and '-' only
#   EXTRA_ARGS        [empty]  extra driver arguments after --arm, split on
#                     whitespace (no quoting); needs ARM_SUFFIX
#   OUTPUT_ROOT       [outputs/sparknet-grow-dendrites-v3]
#   DATA_CONFIG       [configs/data/speech_commands_v2_mfcc32_paper.yaml]
#   TRAIN_CONFIG      [configs/train/sparknet_grow_dendrites_paper.yaml]
#   SWITCH_EPOCH      passed as --switch-epoch only when set; needs ARM_SUFFIX
#   CANDIDATE_EPOCHS  passed as --candidate-epochs only when set; needs ARM_SUFFIX
#   MAX_MINUTES       passed as --max-minutes only when set
#   MAX_TOTAL_MINUTES [720] aggregate worst-case budget: MAX_MINUTES multiplied
#                     by planned-to-run count must fit; set 0 to opt out
#   DRY_RUN           [0]  1 = print the plan only
#   GROW_CMD          [uv run --env-file .env python -m kws.optimize.sparknet_grow_dendrites]
#                     testing hook: the driver command, split on whitespace
#                     (no quoting), so the loop can be exercised against a stub
#
# Each run writes to ${OUTPUT_ROOT}/${arm}/c${width}-seed${seed} and tees its
# stdout+stderr to ${OUTPUT_ROOT}/${arm}/c${width}-seed${seed}.log, where
# arm=${placement}${ARM_SUFFIX} (so the default arm's directories are the
# placement's, as before arms existed).
#
# RESUMING
#
# Re-running is safe: a combination whose reports/grow_summary.yaml says
# `status: complete` (for the same arm, see ARMS) is skipped.  The driver
# cannot resume a run, and it refuses (exit 2) to start in a directory that
# already holds an earlier attempt, so a combination whose output directory
# exists but is not complete
# -- killed, crashed, or stopped by MAX_MINUTES -- is moved aside to
# <output_dir>.partial-<UTC timestamp> (its log to the same name plus .log) and
# restarted from epoch 0.  Nothing is ever deleted.  Two consequences: a
# MAX_MINUTES cap below ~13 min means a run can never finish, and two sweeps
# must not be pointed at the same combination at once, because the second would
# move the first's live run aside.  To redo a *completed* run, move its
# directory away by hand first.
#
# Ctrl-C stops the current run and the sweep, and still prints the summary.
#
# AFTERWARDS
#
# The paired grow-vs-control comparison and report are produced by
# scripts/report_sparknet_grow.py (written separately), which reads this
# sweep's OUTPUT_ROOT and the control directories above.
# scripts/diagnose_grow_run.py --run-dir <run> re-evaluates one finished run's
# clean exports without PerforatedAI and writes reports/grow_diagnostics.yaml.
#
# Exits non-zero if any run did not complete, after attempting all of them
# (130 if interrupted).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WIDTHS="${WIDTHS:-8 12 4 10 6 2 16}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PLACEMENTS="${PLACEMENTS:-fc pointwise}"
ARM_SUFFIX="${ARM_SUFFIX:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparknet-grow-dendrites-v3}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/speech_commands_v2_mfcc32_paper.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/sparknet_grow_dendrites_paper.yaml}"
SWITCH_EPOCH="${SWITCH_EPOCH:-}"
CANDIDATE_EPOCHS="${CANDIDATE_EPOCHS:-}"
MAX_MINUTES="${MAX_MINUTES:-}"
MAX_TOTAL_MINUTES="${MAX_TOTAL_MINUTES:-720}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_GROW_CMD="uv run --env-file .env python -m kws.optimize.sparknet_grow_dendrites"
GROW_CMD="${GROW_CMD:-$DEFAULT_GROW_CMD}"

read -r -a grow_cmd <<<"$GROW_CMD"
read -r -a extra_args <<<"$EXTRA_ARGS"

dry_run=0
if [[ "$DRY_RUN" == "1" ]]; then
  dry_run=1
fi

# A recipe variant must run under its own arm.  In the default arm's
# directories it would either be skipped (the default run there is already
# complete) or become a run that looks like the default arm's.  Refused even in
# a dry run: a plan for the wrong directories is worse than no plan.
if [[ -z "$ARM_SUFFIX" ]]; then
  variant_settings=()
  if ((${#extra_args[@]} > 0)); then
    variant_settings+=("EXTRA_ARGS=\"${EXTRA_ARGS}\"")
  fi
  if [[ -n "$SWITCH_EPOCH" ]]; then
    variant_settings+=("SWITCH_EPOCH=${SWITCH_EPOCH}")
  fi
  if [[ -n "$CANDIDATE_EPOCHS" ]]; then
    variant_settings+=("CANDIDATE_EPOCHS=${CANDIDATE_EPOCHS}")
  fi
  if ((${#variant_settings[@]} > 0)); then
    echo "refusing to start: ${variant_settings[*]} changes the recipe, but ARM_SUFFIX is empty." >&2
    echo "Name the variant's arm, e.g. ARM_SUFFIX=-switch170, so its runs go to" >&2
    echo "${OUTPUT_ROOT}/<placement><ARM_SUFFIX>/ and never mix with the default arm." >&2
    exit 1
  fi
fi

# Pre-flight.  A typo in a placement or a missing config would otherwise
# surface as dozens of identical failures spread over the run; catch it before
# anything starts.  In a dry run the problems are reported but the plan is
# still printed, since the point of a dry run is to see it.
problems=()
if [[ "$GROW_CMD" == "$DEFAULT_GROW_CMD" ]] &&
  [[ ! -f src/kws/optimize/sparknet_grow_dendrites.py ]] &&
  [[ ! -d src/kws/optimize/sparknet_grow_dendrites ]]; then
  problems+=("missing driver: src/kws/optimize/sparknet_grow_dendrites.py")
fi
for file in "$DATA_CONFIG" "$TRAIN_CONFIG"; do
  [[ -f "$file" ]] || problems+=("missing config: ${file}")
done
for width in $WIDTHS; do
  model_token="$width"
  [[ "$model_token" == c* ]] || model_token="c${model_token}"
  model_config="configs/model/sparknet_${model_token}_paper.yaml"
  [[ -f "$model_config" ]] || problems+=("missing config: ${model_config}")
done
if [[ -f "$TRAIN_CONFIG" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    config_python=("$REPO_ROOT/.venv/bin/python")
  else
    config_python=(uv run --env-file .env python)
  fi
  placement_problems="$("${config_python[@]}" - "$TRAIN_CONFIG" "$PLACEMENTS" <<'PY'
import sys

import yaml

path, requested = sys.argv[1:]
try:
    with open(path) as stream:
        config = yaml.safe_load(stream)
    grow = config.get("grow_dendrites") if isinstance(config, dict) else None
    placements = grow.get("placements") if isinstance(grow, dict) else None
    if not isinstance(placements, dict) or not placements:
        print(f"invalid grow_dendrites.placements mapping in config: {path}")
    else:
        configured = ", ".join(sorted(str(name) for name in placements))
        for name in requested.split():
            if name not in placements:
                print(f"unknown placement: {name} (configured in {path}: {configured})")
except Exception as exc:
    print(f"could not read placements from {path}: {type(exc).__name__}: {exc}")
PY
)"
  while IFS= read -r problem; do
    [[ -z "$problem" ]] || problems+=("$problem")
  done <<< "$placement_problems"
fi
# The arm is a directory name and the driver's --arm label.
arm_suffix_pattern='^[A-Za-z0-9._-]*$'
if [[ ! $ARM_SUFFIX =~ $arm_suffix_pattern ]]; then
  problems+=("invalid ARM_SUFFIX: '${ARM_SUFFIX}' (letters, digits, '.', '_' and '-' only)")
fi
if ((${#problems[@]} > 0)); then
  for problem in "${problems[@]}"; do
    echo "PREFLIGHT  ${problem}" >&2
  done
  if ((dry_run == 0)); then
    echo "refusing to start the sweep" >&2
    exit 1
  fi
  echo "(dry run: continuing so the plan can be inspected)" >&2
fi

# Completion marker.  The driver writes reports/grow_summary.yaml both when it
# finishes and when the wall-clock cap stops it, so the file existing is not
# enough; only a top-level `status: complete` line means the run is done.
summary_file() {
  echo "$1/reports/grow_summary.yaml"
}
run_is_complete() {
  local summary
  summary="$(summary_file "$1")"
  [[ -f "$summary" ]] && grep -qx 'status: complete' "$summary"
}
# A top-level scalar of a summary, unquoted; empty when absent or null.
summary_value() {
  local line value
  line="$(grep -m1 "^$2:" "$1" || true)"
  value="${line#"$2":}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  value="${value#\'}"
  value="${value%\'}"
  value="${value#\"}"
  value="${value%\"}"
  case "$value" in
    null | "~") value="" ;;
  esac
  echo "$value"
}
# The arm a run directory's summary records.  Summaries written before the
# driver recorded arms have none; their arm is their placement.
recorded_arm() {
  local summary arm
  summary="$(summary_file "$1")"
  arm="$(summary_value "$summary" arm)"
  if [[ -z "$arm" ]]; then
    arm="$(summary_value "$summary" placement)"
  fi
  echo "$arm"
}

if [[ -n "$MAX_MINUTES" && "$MAX_TOTAL_MINUTES" != "0" ]]; then
  number_pattern='^[0-9]+([.][0-9]+)?$'
  if [[ ! "$MAX_MINUTES" =~ $number_pattern || ! "$MAX_TOTAL_MINUTES" =~ $number_pattern ]]; then
    echo "refusing to start: MAX_MINUTES and MAX_TOTAL_MINUTES must be non-negative numbers" >&2
    exit 1
  fi
  planned_count=0
  for seed in $SEEDS; do
    for width in $WIDTHS; do
      for placement in $PLACEMENTS; do
        token="$width"; [[ "$token" == c* ]] || token="c${token}"
        output_dir="${OUTPUT_ROOT}/${placement}${ARM_SUFFIX}/${token}-seed${seed}"
        if run_is_complete "$output_dir" && [[ "$(recorded_arm "$output_dir")" == "${placement}${ARM_SUFFIX}" ]]; then
          continue
        fi
        planned_count=$((planned_count + 1))
      done
    done
  done
  worst_case="$(awk -v minutes="$MAX_MINUTES" -v count="$planned_count" 'BEGIN { print minutes * count }')"
  if awk -v worst="$worst_case" -v total="$MAX_TOTAL_MINUTES" \
    'BEGIN { exit !(worst > total + 1e-9) }'; then
    echo "refusing to start: aggregate budget ${worst_case} minutes exceeds MAX_TOTAL_MINUTES=${MAX_TOTAL_MINUTES} (${MAX_MINUTES} x ${planned_count} planned-to-run)" >&2
    exit 1
  fi
fi

# What an unfinished directory looked like, for the move-aside message.
describe_partial() {
  local summary line
  summary="$(summary_file "$1")"
  if [[ -f "$summary" ]]; then
    line="$(grep -m1 '^status:' "$summary" || true)"
    echo "grow_summary ${line:-has no status line}"
  else
    echo "no grow_summary.yaml"
  fi
}

format_duration() {
  printf '%dm%02ds' $(($1 / 60)) $(($1 % 60))
}

# Ctrl-C reaches the driver through the terminal's process group either way;
# trapping it here keeps this script alive long enough to stop the loop and
# print the summary, instead of treating the killed run as an ordinary failure
# and moving on to the next one.
interrupted=0
trap 'interrupted=1' INT

statuses=()
failures=0

arm_width=3
arm_names=""
for placement in $PLACEMENTS; do
  arm="${placement}${ARM_SUFFIX}"
  arm_names="${arm_names:+${arm_names} }${arm}"
  if ((${#arm} > arm_width)); then
    arm_width=${#arm}
  fi
done
echo "arms: ${arm_names}  (driver extra args: ${EXTRA_ARGS:-none})"

if ((dry_run == 0)); then
  mkdir -p "$OUTPUT_ROOT"
fi

for seed in $SEEDS; do
  for width in $WIDTHS; do
    for placement in $PLACEMENTS; do
      if ((interrupted)); then
        break 3
      fi

      arm="${placement}${ARM_SUFFIX}"
      model_token="$width"; [[ "$model_token" == c* ]] || model_token="c${model_token}"
      label="${arm} C${model_token#c} seed${seed}"
      output_dir="${OUTPUT_ROOT}/${arm}/${model_token}-seed${seed}"
      log_file="${output_dir}.log"

      cmd=("${grow_cmd[@]}"
        --data-config "$DATA_CONFIG"
        --model-config "configs/model/sparknet_${model_token}_paper.yaml"
        --train-config "$TRAIN_CONFIG"
        --placement "$placement" --seed "$seed"
        --output-dir "$output_dir")
      [[ -n "$SWITCH_EPOCH" ]] && cmd+=(--switch-epoch "$SWITCH_EPOCH")
      [[ -n "$CANDIDATE_EPOCHS" ]] && cmd+=(--candidate-epochs "$CANDIDATE_EPOCHS")
      [[ -n "$MAX_MINUTES" ]] && cmd+=(--max-minutes "$MAX_MINUTES")
      cmd+=(--arm "$arm" ${extra_args[@]+"${extra_args[@]}"})

      if run_is_complete "$output_dir"; then
        recorded="$(recorded_arm "$output_dir")"
        if [[ "$recorded" != "$arm" ]]; then
          # Skipping would pass another arm's result off as this one, and
          # moving it aside would bury a finished run.  Leave it alone.
          mismatch="${output_dir} is a complete run of arm '${recorded:-?}', not '${arm}'"
          if ((dry_run)); then
            echo "refuse  ${label}  ${mismatch}; would neither skip nor replace it" >&2
            statuses+=("${arm}|${width}|${seed}|would-refuse(arm=${recorded:-?})|-")
          else
            echo "FAIL  ${label}  ${mismatch}; refusing to skip or replace it" >&2
            statuses+=("${arm}|${width}|${seed}|arm-mismatch(${recorded:-?})|-")
            failures=$((failures + 1))
          fi
          continue
        fi
        echo "skip  ${label}  (${output_dir} already complete)"
        if ((dry_run)); then
          statuses+=("${arm}|${width}|${seed}|would-skip|-")
        else
          statuses+=("${arm}|${width}|${seed}|skipped|-")
        fi
        continue
      fi

      # An earlier attempt the driver would refuse to start over.  Move it --
      # and its log -- aside under one timestamp so the pair stays together and
      # the failed attempt stays diagnosable.
      prefix=""
      if [[ -e "$output_dir" || -e "$log_file" ]]; then
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        aside="${output_dir}.partial-${stamp}"
        n=1
        while [[ -e "$aside" || -e "${aside}.log" ]]; do
          aside="${output_dir}.partial-${stamp}-${n}"
          n=$((n + 1))
        done
        if [[ -e "$output_dir" ]]; then
          prefix="moved-aside+"
          echo "move  ${label}  ${output_dir} is not complete ($(describe_partial "$output_dir"))"
          echo "      dir -> ${aside}"
        else
          echo "move  ${label}  log of an earlier attempt that left no output dir"
        fi
        if [[ -e "$log_file" ]]; then
          echo "      log -> ${aside}.log"
        fi
        if ((dry_run == 0)); then
          moved=1
          if [[ -e "$output_dir" ]]; then
            mv "$output_dir" "$aside" || moved=0
          fi
          if ((moved)) && [[ -e "$log_file" ]]; then
            mv "$log_file" "${aside}.log" || moved=0
          fi
          if ((moved == 0)); then
            echo "FAIL  ${label}  could not move the earlier attempt aside; not starting" >&2
            statuses+=("${arm}|${width}|${seed}|move-failed|-")
            failures=$((failures + 1))
            continue
          fi
        fi
      fi

      echo "run   ${label}  -> ${output_dir}  (log: ${log_file})"
      if ((dry_run)); then
        printf '      '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        statuses+=("${arm}|${width}|${seed}|would-${prefix:+move-aside+}run|-")
        continue
      fi

      mkdir -p "$(dirname "$output_dir")"
      started=$SECONDS
      # The pipeline's own status would be tee's under a plain `if`, and with
      # pipefail a non-zero driver exit would trip `set -e`.  So run it with
      # errexit off and read the driver's exit code straight out of PIPESTATUS.
      set +e
      "${cmd[@]}" 2>&1 | tee "$log_file"
      pipe_status=("${PIPESTATUS[@]}")
      set -e
      rc="${pipe_status[0]}"
      tee_rc="${pipe_status[1]}"
      elapsed="$(format_duration $((SECONDS - started)))"

      if ((interrupted)); then
        echo "STOP  ${label}  interrupted (driver exit ${rc}); stopping the sweep" >&2
        statuses+=("${arm}|${width}|${seed}|${prefix}interrupted|${elapsed}")
        failures=$((failures + 1))
        break 3
      fi
      if ((tee_rc != 0)); then
        echo "WARN  ${label}  tee exited ${tee_rc}; ${log_file} may be incomplete" >&2
      fi

      case "$rc" in
        0)
          if run_is_complete "$output_dir"; then
            echo "done  ${label}  in ${elapsed}"
            statuses+=("${arm}|${width}|${seed}|${prefix}ok|${elapsed}")
          else
            # Exit 0 without the marker a finished run leaves.  Count it as a
            # failure rather than letting the next invocation quietly move it
            # aside and re-run it as if nothing had happened.
            echo "WARN  ${label}  exited 0 but $(summary_file "$output_dir") does not say 'status: complete'" >&2
            statuses+=("${arm}|${width}|${seed}|${prefix}incomplete|${elapsed}")
            failures=$((failures + 1))
          fi
          ;;
        3)
          echo "CAP   ${label}  stopped by the wall-clock cap${MAX_MINUTES:+ (--max-minutes ${MAX_MINUTES})} after ${elapsed} (see ${log_file})" >&2
          statuses+=("${arm}|${width}|${seed}|${prefix}capped|${elapsed}")
          failures=$((failures + 1))
          ;;
        2)
          # Should not happen: anything the driver would refuse was moved
          # aside above.  Reaching it means the refusal check and the
          # move-aside logic disagree about what an earlier attempt looks like.
          echo "FAIL  ${label}  driver refused to start in ${output_dir} (exit 2; see ${log_file})" >&2
          statuses+=("${arm}|${width}|${seed}|${prefix}refused|${elapsed}")
          failures=$((failures + 1))
          ;;
        *)
          # One bad run must not cost the rest of the grid, so record it and
          # keep going; the exit status at the end reports it.
          echo "FAIL  ${label}  exit ${rc} (see ${log_file})" >&2
          statuses+=("${arm}|${width}|${seed}|${prefix}failed(exit=${rc})|${elapsed}")
          failures=$((failures + 1))
          ;;
      esac
    done
  done
done

trap - INT

echo
if ((dry_run)); then
  echo "sweep plan, DRY_RUN (${OUTPUT_ROOT})"
else
  echo "sweep summary (${OUTPUT_ROOT})"
fi
row_format="%-${arm_width}s  %-6s %-5s %-9s %s\n"
# shellcheck disable=SC2059  # the format carries the arm column's width
printf "$row_format" "arm" "width" "seed" "time" "status"
# shellcheck disable=SC2059
printf "$row_format" "---" "-----" "----" "----" "------"
for entry in ${statuses[@]+"${statuses[@]}"}; do
  IFS='|' read -r arm width seed status elapsed <<<"$entry"
  # shellcheck disable=SC2059
  printf "$row_format" "$arm" "C${width#c}" "$seed" "$elapsed" "$status"
done

if ((interrupted)); then
  echo
  echo "sweep interrupted; ${failures} run(s) did not complete" >&2
  exit 130
fi
if ((failures > 0)); then
  echo
  echo "${failures} run(s) did not complete" >&2
  exit 1
fi
