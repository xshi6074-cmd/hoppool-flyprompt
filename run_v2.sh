#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# FlyGCL queued experiment runner
#
# Daily usage:
#   1. Edit only the USER CONFIG section below.
#   2. Preview: bash run_v2.sh --dry-run
#   3. Launch:  bash run_v2.sh
#   4. Debug in foreground: bash run_v2.sh --foreground
#
# By default the scheduler runs inside one detached master screen. It expands
# every task seed into an independent job and keeps at most one job on each GPU.
# When a GPU becomes free, it receives the next queued job.
# =============================================================================

declare -a TASK_NAMES=()
declare -a TASK_SEED_SPECS=()
declare -a TASK_ARG_VARS=()

die() {
    echo "[run_v2] ERROR: $*" >&2
    exit 1
}

add_task() {
    if (( $# < 3 )); then
        die "add_task requires: <name> \"<seed list>\" <main.py args...>"
    fi

    local task_name=$1
    local seed_spec=$2
    shift 2

    if [[ ! "$task_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
        die "Invalid task name '$task_name'; use letters, numbers, dot, dash or underscore"
    fi

    local existing_name
    for existing_name in "${TASK_NAMES[@]}"; do
        if [[ "$existing_name" == "$task_name" ]]; then
            die "Duplicate task name: $task_name"
        fi
    done

    local task_index=${#TASK_NAMES[@]}
    local args_var="TASK_ARGS_${task_index}"
    declare -g -a "${args_var}=()"
    local -n task_args_ref="$args_var"
    task_args_ref=("$@")

    TASK_NAMES+=("$task_name")
    TASK_SEED_SPECS+=("$seed_spec")
    TASK_ARG_VARS+=("$args_var")
}


# =============================================================================
# USER CONFIG
# Edit this section for each experiment batch. Do not edit scheduler code below.
# =============================================================================

RUN_NAME="generator_teacher_distill_gradnorm"
GPU_POOL=(0 1 2 3 4 5 6 7)

BACKBONE_PATH="/data2/hongwei/.cache/torch/hub/checkpoints/ViT-B_16.npz"
CIFAR_DATA_DIR="/data/datasets/cifar-100-python"
CUB_DATA_DIR="/data/datasets/CUB_200_2011"

# Scheduler/runtime settings.
POLL_SECONDS=5
RUN_ROOT="./results/run_v2"
PYTHON_BIN="${PYTHON:-python}"
STRICT_PATHS=true

# Limit CPU thread pools for every Python training process.
CPU_THREADS=1

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export BLIS_NUM_THREADS="$CPU_THREADS"

# Do not let OpenMP workers busy-spin while waiting.
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

# These arguments are prepended to every task. Arguments written in add_task
# appear later and therefore may override the same argparse option.
COMMON_ARGS=(
    --method flyprompt
    --backbone vit_base_patch16_224
    --backbone_path "$BACKBONE_PATH"
    --n_tasks 5
    --step_num 10
    --n 50
    --m 10
    --rnd_NM
    --opt_name adam
    --lr 0.005
    --sched_name default
    --num_epochs 3
    --batchsize 64
    --online_iter 3
    --transforms autoaug
    --topk 1
    --eval_period 1000
    --use_amp
    --n_worker 0
)

# -----------------------------------------------------------------------------
# MLP-generator controlled ablations
#
# One shared default is defined in COMMON_ARGS. Each task below changes exactly
# one factor. The baseline is listed once per dataset rather than forming a
# Cartesian grid.
# -----------------------------------------------------------------------------

# add_generator_ablation() {
#     local task_name=$1
#     shift
#     add_task "$task_name" "1 2 3" "$@"
# }

# # Baseline: shallow block 0, one Linear layer, one insertion point, MSE,
# # diagonal statistics, lag-1 teacher, eight replay classes.
# add_generator_ablation "baseline_b0_mlp1_mse_diag_t1_r8" \
#     --prompt_blocks 0

# # Single prompt insertion location: shallow / middle / deep.
# add_generator_ablation "insert_mid_b5" \
#     --prompt_blocks 5
# add_generator_ablation "insert_deep_b11" \
#     --prompt_blocks 11

# # Generator depth while prompt insertion remains at block 0 only.
# add_generator_ablation "mlp_depth_2" \
#     --generator_layers 2
# add_generator_ablation "mlp_depth_3" \
#     --generator_layers 3

# # Number of early insertion points while the generator stays one Linear layer.
# add_generator_ablation "prompt_points_3_b0-2" \
#     --prompt_blocks 0 1 2
# add_generator_ablation "prompt_points_5_b0-4" \
#     --prompt_blocks 0 1 2 3 4

# # Distillation distance.
# add_generator_ablation "loss_cosine" \
#     --distill_metric cosine

# # Replay statistics: rank-16 correlated component plus residual diagonal.
# add_generator_ablation "statistics_low_rank_r16" \
#     --statistic_type low_rank \
#     --covariance_rank 16

# # Hard teachers: previous internal step plus the snapshot five steps back.
# # The total replay-class budget remains eight and is split across active teachers.
# add_generator_ablation "teachers_lag1_lag5" \
#     --teacher_lags 1 5

# # Total old-class replay budget per optimizer step.
# add_generator_ablation "replay_classes_4" \
#     --replay_class_budget 4
# add_generator_ablation "replay_classes_16" \
#     --replay_class_budget 16

# # The same 12 controlled MLP-generator configurations on CUB-200.
# add_cub_generator_ablation() {
#     local task_name=$1
#     shift
#     add_task "cub_${task_name}" "1 2 3" \
#         --dataset cub200 \
#         --data_dir "$CUB_DATA_DIR" \
#         "$@"
# }

# add_cub_generator_ablation "baseline_b0_mlp1_mse_diag_t1_r8" \
#     --prompt_blocks 0
# add_cub_generator_ablation "insert_mid_b5" \
#     --prompt_blocks 5
# add_cub_generator_ablation "insert_deep_b11" \
#     --prompt_blocks 11
# add_cub_generator_ablation "mlp_depth_2" \
#     --generator_layers 2
# add_cub_generator_ablation "mlp_depth_3" \
#     --generator_layers 3
# add_cub_generator_ablation "prompt_points_3_b0-2" \
#     --prompt_blocks 0 1 2
# add_cub_generator_ablation "prompt_points_5_b0-4" \
#     --prompt_blocks 0 1 2 3 4
# add_cub_generator_ablation "loss_cosine" \
#     --distill_metric cosine
# add_cub_generator_ablation "statistics_low_rank_r16" \
#     --statistic_type low_rank \
#     --covariance_rank 16
# add_cub_generator_ablation "teachers_lag1_lag5" \
#     --teacher_lags 1 5
# add_cub_generator_ablation "replay_classes_4" \
#     --replay_class_budget 4
# add_cub_generator_ablation "replay_classes_16" \
#     --replay_class_budget 16

# Mechanism ablation. Keep the output projection trainable in every HFPool arm,
# then add key-side routing, pooling-query routing, or both. The ViT gate arm
# scales each frozen attention head independently in the five prompt layers.
add_attention_gate_suite() {
    local prefix=$1
    local dataset=$2
    local data_dir=$3

    local -a dataset_args=(
        --dataset "$dataset"
        --data_dir "$data_dir"
    )
    local -a prompt_layout=(
        --deep_prompts 0
        --num_pooling_heads 1
        --num_pooling_blocks 3
        --prompt_length 10
    )
    local -a hopfield_clip=(
        --hopfield_grad_clip
        --hopfield_grad_clip_norm 1.0
    )

    add_task "${prefix}_vit_head_gate" "1 2 3" \
        "${dataset_args[@]}" \
        --method gate \
        --gate_blocks 0 1 2 3 4

    # The 2026-08-31 batch also trained K at 0.1x LR (--hopfield_qk_lr_scale),
    # which no longer exists; these arms keep only gradient clipping.
    local -a trainable_sets=("o k" "o query" "o k query")
    local trainable
    for trainable in "${trainable_sets[@]}"; do
        add_task "${prefix}_hfpool_${trainable// /_}" "1 2 3" \
            "${dataset_args[@]}" \
            "${prompt_layout[@]}" \
            "${hopfield_clip[@]}" \
            --method hfpool \
            --hopfield_trainable $trainable
    done
}

# add_attention_gate_suite "cifar100" "cifar100" "$CIFAR_DATA_DIR"
# add_attention_gate_suite "cub200" "cub200" "$CUB_DATA_DIR"

# -----------------------------------------------------------------------------
# MLP-generator teacher-distillation suite
#
# Question: can hard-teacher distillation remain numerically controlled and
# contribute after a short post-snapshot CE-only adaptation period?
#
# All four arms use one-sided GradNorm-lite. CE stays at weight 1 and the
# detached distillation weight targets a gradient-norm ratio of 0.25 on the
# generator's final Linear weight. Every hard snapshot resets both the 100
# stream-sample delay and the controller EMA. With batch size 64, whole-batch
# gating skips the first two post-snapshot batches (128 actual samples).
#
# The six distill-off jobs from
# generator_attention_gate_ablation_20260901_150057 are a provisional,
# cross-commit reference only. This runner intentionally does not rerun them.
#
# Every arm uses --replay_eligibility previous: a teacher reads its eligible
# classes from the snapshot one internal step older than itself, so a class
# introduced while that teacher was still learning it is not replayed.
# -----------------------------------------------------------------------------

add_generator_distill_suite() {
    local prefix=$1
    local dataset=$2
    local data_dir=$3

    local -a generator_base=(
        --method mlp_generator
        --dataset "$dataset"
        --data_dir "$data_dir"
        --len_prompt 10
        --prompt_blocks 0
        --generator_layers 1
        --teacher_lags 1
        --replay_class_budget 8
        --min_replay_samples 2
        --replay_eligibility previous
        --distill_weight 1.0
        --distill_weight_mode gradnorm
        --distill_delay_samples 100
        --distill_grad_ratio 0.25
        --distill_grad_ema 0.9
        --distill_weight_min 1e-4
        --distill_weight_max 1e4
    )

    local metric
    for metric in mse cosine; do
        add_task "${prefix}_gen_sample_diag_raw_${metric}_gradnorm" "1 2 3" \
            "${generator_base[@]}" \
            --distill_metric "$metric" \
            --distill_mode sample \
            --statistic_type diagonal \
            --statistics_space raw
    done

    add_task "${prefix}_gen_closed_form_r16_mse_gradnorm" "1 2 3" \
        "${generator_base[@]}" \
        --distill_mode closed_form \
        --distill_metric mse \
        --statistic_type low_rank \
        --covariance_rank 16 \
        --statistics_space normalized

    add_task "${prefix}_gen_sample_r16_k8_cosine_gradnorm" "1 2 3" \
        "${generator_base[@]}" \
        --distill_mode sample \
        --distill_metric cosine \
        --statistic_type low_rank \
        --covariance_rank 16 \
        --statistics_space normalized \
        --anchors_per_class 8
}

add_generator_distill_suite "cifar100" "cifar100" "$CIFAR_DATA_DIR"
add_generator_distill_suite "cub200" "cub200" "$CUB_DATA_DIR"

# =============================================================================
# SCHEDULER IMPLEMENTATION
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

MODE="detach"
INTERNAL_WORKER=false
REQUESTED_RUN_ID=""
SELF_TEST=false

usage() {
    cat <<'EOF'
Usage:
  bash run_v2.sh                 Launch scheduler in a detached master screen
  bash run_v2.sh --dry-run       Validate config structure and print all jobs
  bash run_v2.sh --foreground    Run scheduler in the current terminal
  bash run_v2.sh --self-test     Test queue control without Python or GPUs
  bash run_v2.sh --help          Show this help

Experiment definitions live in the USER CONFIG section of run_v2.sh.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            MODE="dry-run"
            shift
            ;;
        --foreground)
            MODE="foreground"
            shift
            ;;
        --self-test)
            MODE="foreground"
            SELF_TEST=true
            shift
            ;;
        --worker)
            (( $# >= 2 )) || die "--worker requires an internal run id"
            INTERNAL_WORKER=true
            MODE="foreground"
            REQUESTED_RUN_ID=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "Unknown runner argument: $1"
            ;;
    esac
done

if [[ "$SELF_TEST" == true ]]; then
    command -v mktemp >/dev/null 2>&1 || die "'mktemp' is required for --self-test"
    GPU_POOL=(0 1)
    POLL_SECONDS=1
    RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/run_v2-self-test.XXXXXX")"
    # Git Bash's native mktemp may print a Windows drive path even when given
    # /tmp. Normalize it before the POSIX absolute-path check below.
    case "$RUN_ROOT" in
        [A-Za-z]:/*|[A-Za-z]:\\*)
            windows_root=${RUN_ROOT//\\//}
            drive_letter=${windows_root:0:1}
            RUN_ROOT="/${drive_letter,,}${windows_root:2}"
            ;;
    esac
    PYTHON_BIN=true
    STRICT_PATHS=false
fi

sanitize_name() {
    local value=$1
    value=${value//[^A-Za-z0-9._-]/_}
    printf '%s' "${value:0:96}"
}

command_exists() {
    if [[ "$1" == */* ]]; then
        [[ -x "$1" ]]
    else
        command -v "$1" >/dev/null 2>&1
    fi
}

option_value() {
    local target=$1
    shift
    OPTION_VALUE=""

    local -a values=("$@")
    local index=0
    local arg
    while (( index < ${#values[@]} )); do
        arg=${values[$index]}
        if [[ "$arg" == "$target" ]]; then
            if (( index + 1 >= ${#values[@]} )); then
                die "$target is missing its value"
            fi
            OPTION_VALUE=${values[$((index + 1))]}
            ((index += 2))
            continue
        fi
        if [[ "$arg" == "${target}="* ]]; then
            OPTION_VALUE=${arg#*=}
        fi
        ((index += 1))
    done
}

validate_no_reserved_args() {
    local scope=$1
    shift

    local arg
    for arg in "$@"; do
        case "$arg" in
            --gpu|--gpu=*|--seeds|--seeds=*)
                die "$scope must not set $arg; GPU and seed are managed by run_v2"
                ;;
        esac
    done
}

declare -a JOB_TASK_INDEX=()
declare -a JOB_SEEDS=()
declare -a JOB_IDS=()

validate_and_expand_config() {
    [[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] \
        || die "RUN_NAME must use letters, numbers, dot, dash or underscore"
    (( ${#GPU_POOL[@]} > 0 )) || die "GPU_POOL is empty"
    (( ${#TASK_NAMES[@]} > 0 )) || die "No tasks configured"
    [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] \
        || die "POLL_SECONDS must be a positive integer"
    [[ "$STRICT_PATHS" == true || "$STRICT_PATHS" == false ]] \
        || die "STRICT_PATHS must be true or false"
    [[ -n "$PYTHON_BIN" ]] || die "PYTHON_BIN is empty"

    local -A seen_gpus=()
    local gpu
    for gpu in "${GPU_POOL[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu"
        [[ -z "${seen_gpus[$gpu]:-}" ]] || die "Duplicate GPU id: $gpu"
        seen_gpus[$gpu]=1
    done

    validate_no_reserved_args "COMMON_ARGS" "${COMMON_ARGS[@]}"

    local task_index
    for task_index in "${!TASK_NAMES[@]}"; do
        local args_var=${TASK_ARG_VARS[$task_index]}
        local -n task_args_ref="$args_var"
        validate_no_reserved_args \
            "task '${TASK_NAMES[$task_index]}'" "${task_args_ref[@]}"

        local -a effective_args=("${COMMON_ARGS[@]}" "${task_args_ref[@]}")
        option_value "--method" "${effective_args[@]}"
        [[ -n "$OPTION_VALUE" ]] \
            || die "Task '${TASK_NAMES[$task_index]}' is missing --method"
        option_value "--dataset" "${effective_args[@]}"
        [[ -n "$OPTION_VALUE" ]] \
            || die "Task '${TASK_NAMES[$task_index]}' is missing --dataset"
        option_value "--backbone" "${effective_args[@]}"
        [[ -n "$OPTION_VALUE" ]] \
            || die "Task '${TASK_NAMES[$task_index]}' is missing --backbone"

        local normalized_seeds=${TASK_SEED_SPECS[$task_index]//,/ }
        local -a task_seeds=()
        read -r -a task_seeds <<< "$normalized_seeds"
        (( ${#task_seeds[@]} > 0 )) \
            || die "Task '${TASK_NAMES[$task_index]}' has no seeds"

        local -A seen_seeds=()
        local seed
        for seed in "${task_seeds[@]}"; do
            [[ "$seed" =~ ^-?[0-9]+$ ]] \
                || die "Task '${TASK_NAMES[$task_index]}' has invalid seed '$seed'"
            [[ -z "${seen_seeds[$seed]:-}" ]] \
                || die "Task '${TASK_NAMES[$task_index]}' repeats seed '$seed'"
            seen_seeds[$seed]=1

            local job_number=$((${#JOB_IDS[@]} + 1))
            local job_id
            printf -v job_id "job_%03d_%s_seed%s" \
                "$job_number" "${TASK_NAMES[$task_index]}" "$seed"
            JOB_TASK_INDEX+=("$task_index")
            JOB_SEEDS+=("$seed")
            JOB_IDS+=("$job_id")
        done
    done
}

validate_paths_for_task() {
    local task_index=$1
    [[ "$STRICT_PATHS" == true ]] || return 0

    local args_var=${TASK_ARG_VARS[$task_index]}
    local -n task_args_ref="$args_var"
    local -a effective_args=("${COMMON_ARGS[@]}" "${task_args_ref[@]}")

    option_value "--backbone_path" "${effective_args[@]}"
    if [[ -n "$OPTION_VALUE" ]]; then
        [[ -f "$OPTION_VALUE" && -r "$OPTION_VALUE" ]] \
            || die "Checkpoint is not a readable file: $OPTION_VALUE"
    fi

    option_value "--data_dir" "${effective_args[@]}"
    if [[ -n "$OPTION_VALUE" ]]; then
        [[ -d "$OPTION_VALUE" && -r "$OPTION_VALUE" ]] \
            || die "Dataset directory is not readable: $OPTION_VALUE"
    fi
}

quote_command() {
    local -a command_parts=("$@")
    printf -v QUOTED_COMMAND '%q ' "${command_parts[@]}"
    QUOTED_COMMAND=${QUOTED_COMMAND% }
}

build_job_command() {
    local job_index=$1
    local gpu=$2
    local task_index=${JOB_TASK_INDEX[$job_index]}
    local task_name=${TASK_NAMES[$task_index]}
    local seed=${JOB_SEEDS[$job_index]}
    local args_var=${TASK_ARG_VARS[$task_index]}
    local -n task_args_ref="$args_var"

    JOB_COMMAND=(
        "$PYTHON_BIN"
        -W ignore
        main.py
        --gpu "$gpu"
        --seeds "$seed"
        --note "${RUN_NAME}_${task_name}"
        "${COMMON_ARGS[@]}"
        "${task_args_ref[@]}"
    )
}

print_plan() {
    echo "============================================================"
    echo "Run name : $RUN_NAME"
    echo "GPU pool : ${GPU_POOL[*]}"
    echo "Tasks    : ${#TASK_NAMES[@]}"
    echo "Jobs     : ${#JOB_IDS[@]}"
    echo "============================================================"

    local job_index
    for job_index in "${!JOB_IDS[@]}"; do
        build_job_command "$job_index" "<dynamic-gpu>"
        quote_command "${JOB_COMMAND[@]}"
        echo
        echo "[$(printf '%03d' "$((job_index + 1))")/${#JOB_IDS[@]}] ${JOB_IDS[$job_index]}"
        echo "  $QUOTED_COMMAND"
    done
    echo
}

validate_and_expand_config

if [[ "$MODE" == "dry-run" ]]; then
    print_plan
    echo "[run_v2] Dry run only; nothing was launched."
    exit 0
fi

command_exists "$PYTHON_BIN" || die "Python interpreter not found: $PYTHON_BIN"

declare -A validated_task_paths=()
for task_index in "${JOB_TASK_INDEX[@]}"; do
    if [[ -z "${validated_task_paths[$task_index]:-}" ]]; then
        validate_paths_for_task "$task_index"
        validated_task_paths[$task_index]=1
    fi
done

timestamp=$(date '+%Y%m%d_%H%M%S')
if [[ -n "$REQUESTED_RUN_ID" ]]; then
    RUN_ID=$REQUESTED_RUN_ID
else
    RUN_ID="$(sanitize_name "$RUN_NAME")_${timestamp}"
fi

if [[ "$MODE" == "detach" && "$INTERNAL_WORKER" == false ]]; then
    command_exists tmux || die "'tmux' is required for detached scheduling"

    master_session="runv2_$(sanitize_name "$RUN_ID")"

    if tmux has-session -t "$master_session" 2>/dev/null; then
        die "Master tmux already exists: $master_session"
    fi

    tmux new-session -d \
        -s "$master_session" \
        "bash \"$SCRIPT_PATH\" --worker \"$RUN_ID\""

    echo "[run_v2] Scheduler launched."
    echo "[run_v2] Master tmux: $master_session"
    echo "[run_v2] Attach: tmux attach -t $master_session"
    echo "[run_v2] Detach without stopping: Ctrl-B, then D"
    exit 0
fi

if [[ "$RUN_ROOT" == /* ]]; then
    RUN_ROOT_ABS=$RUN_ROOT
else
    RUN_ROOT_ABS="${SCRIPT_DIR}/${RUN_ROOT#./}"
fi
RUN_DIR="${RUN_ROOT_ABS}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
STATUS_DIR="${RUN_DIR}/status"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
cp -- "$SCRIPT_PATH" "${RUN_DIR}/run_v2.snapshot.sh"

git_commit=$(git rev-parse HEAD 2>/dev/null || printf 'unknown')
{
    echo "run_id=$RUN_ID"
    echo "run_name=$RUN_NAME"
    echo "git_commit=$git_commit"
    echo "started_at=$(date -Iseconds)"
    echo "gpu_pool=${GPU_POOL[*]}"
    echo "job_count=${#JOB_IDS[@]}"
} > "${RUN_DIR}/run_meta.txt"

printf 'job_id\ttask\tseed\tcommand\n' > "${RUN_DIR}/resolved_jobs.tsv"
for job_index in "${!JOB_IDS[@]}"; do
    build_job_command "$job_index" "<dynamic-gpu>"
    quote_command "${JOB_COMMAND[@]}"
    printf '%s\t%s\t%s\t%s\n' \
        "${JOB_IDS[$job_index]}" \
        "${TASK_NAMES[${JOB_TASK_INDEX[$job_index]}]}" \
        "${JOB_SEEDS[$job_index]}" \
        "$QUOTED_COMMAND" \
        >> "${RUN_DIR}/resolved_jobs.tsv"
done

printf 'event\ttime\tjob_id\tgpu\texit_code\tlog\n' \
    > "${RUN_DIR}/events.tsv"

declare -A ACTIVE_PID=()
declare -A ACTIVE_JOB_INDEX=()
declare -A ACTIVE_STATUS_FILE=()
declare -a FAILED_JOB_IDS=()

PENDING_INDEX=0
COMPLETED_COUNT=0
TOTAL_JOBS=${#JOB_IDS[@]}

stop_active_jobs() {
    local signal=${1:-TERM}
    local active_gpu
    for active_gpu in "${!ACTIVE_PID[@]}"; do
        kill "-$signal" "${ACTIVE_PID[$active_gpu]}" 2>/dev/null || true
    done
}

on_interrupt() {
    echo
    echo "[run_v2] Scheduler interrupted; terminating active jobs..." >&2
    stop_active_jobs TERM
    sleep 1
    stop_active_jobs KILL
    exit 130
}
trap on_interrupt INT TERM

launch_job() {
    local gpu=$1
    local job_index=$2
    local job_id=${JOB_IDS[$job_index]}
    local log_file="${LOG_DIR}/${job_id}.log"
    local status_file="${STATUS_DIR}/${job_id}.status"
    local status_tmp="${status_file}.tmp"

    build_job_command "$job_index" "$gpu"
    quote_command "${JOB_COMMAND[@]}"

    {
        echo "============================================================"
        echo "job_id=$job_id"
        echo "gpu=$gpu"
        echo "seed=${JOB_SEEDS[$job_index]}"
        echo "started_at=$(date -Iseconds)"
        echo "command=$QUOTED_COMMAND"
        echo "============================================================"
    } > "$log_file"

    (
        set +e
        "${JOB_COMMAND[@]}" >> "$log_file" 2>&1
        rc=$?
        printf '%s\n' "$rc" > "$status_tmp"
        mv -f -- "$status_tmp" "$status_file"
        exit "$rc"
    ) &

    local pid=$!
    ACTIVE_PID[$gpu]=$pid
    ACTIVE_JOB_INDEX[$gpu]=$job_index
    ACTIVE_STATUS_FILE[$gpu]=$status_file

    printf 'start\t%s\t%s\t%s\t\t%s\n' \
        "$(date -Iseconds)" "$job_id" "$gpu" "$log_file" \
        >> "${RUN_DIR}/events.tsv"
    echo "[run_v2] START gpu=$gpu job=$job_id log=$log_file"
}

collect_finished_jobs() {
    local gpu
    for gpu in "${GPU_POOL[@]}"; do
        [[ -n "${ACTIVE_PID[$gpu]:-}" ]] || continue

        local status_file=${ACTIVE_STATUS_FILE[$gpu]}
        [[ -f "$status_file" ]] || continue

        local pid=${ACTIVE_PID[$gpu]}
        local job_index=${ACTIVE_JOB_INDEX[$gpu]}
        local job_id=${JOB_IDS[$job_index]}
        local log_file="${LOG_DIR}/${job_id}.log"
        local rc
        read -r rc < "$status_file"

        wait "$pid" 2>/dev/null || true

        printf 'finish\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -Iseconds)" "$job_id" "$gpu" "$rc" "$log_file" \
            >> "${RUN_DIR}/events.tsv"

        if [[ "$rc" == "0" ]]; then
            echo "[run_v2] DONE  gpu=$gpu job=$job_id"
        else
            echo "[run_v2] FAIL  gpu=$gpu job=$job_id rc=$rc log=$log_file" >&2
            FAILED_JOB_IDS+=("$job_id")
        fi

        unset 'ACTIVE_PID[$gpu]'
        unset 'ACTIVE_JOB_INDEX[$gpu]'
        unset 'ACTIVE_STATUS_FILE[$gpu]'
        ((COMPLETED_COUNT += 1))
    done
}

echo "============================================================"
echo "[run_v2] Scheduler started"
echo "[run_v2] Run ID: $RUN_ID"
echo "[run_v2] Git: $git_commit"
echo "[run_v2] GPUs: ${GPU_POOL[*]}"
echo "[run_v2] Jobs: $TOTAL_JOBS"
echo "[run_v2] Artifacts: $RUN_DIR"
echo "============================================================"

while (( COMPLETED_COUNT < TOTAL_JOBS )); do
    collect_finished_jobs

    for gpu in "${GPU_POOL[@]}"; do
        if [[ -z "${ACTIVE_PID[$gpu]:-}" ]] \
                && (( PENDING_INDEX < TOTAL_JOBS )); then
            launch_job "$gpu" "$PENDING_INDEX"
            ((PENDING_INDEX += 1))
        fi
    done

    (( COMPLETED_COUNT >= TOTAL_JOBS )) && break
    sleep "$POLL_SECONDS"
done

trap - INT TERM

{
    echo "finished_at=$(date -Iseconds)"
    echo "completed_jobs=$COMPLETED_COUNT"
    echo "failed_jobs=${#FAILED_JOB_IDS[@]}"
    if (( ${#FAILED_JOB_IDS[@]} > 0 )); then
        echo "failed_job_ids=${FAILED_JOB_IDS[*]}"
    fi
} >> "${RUN_DIR}/run_meta.txt"

echo "============================================================"
echo "[run_v2] All jobs finished: $COMPLETED_COUNT/$TOTAL_JOBS"
echo "[run_v2] Failed jobs: ${#FAILED_JOB_IDS[@]}"
if (( ${#FAILED_JOB_IDS[@]} > 0 )); then
    printf '[run_v2]   %s\n' "${FAILED_JOB_IDS[@]}"
fi
echo "[run_v2] Artifacts: $RUN_DIR"
echo "============================================================"

if (( ${#FAILED_JOB_IDS[@]} > 0 )); then
    exit 1
fi
