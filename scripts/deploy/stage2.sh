#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./scripts/deploy/stage2.sh [options] [-- extra deploy args]

Options:
  --checkpoint PATH        Explicit stage2 checkpoint path
  --use-last-checkpoint    Load latest stage2 model_last.pt instead of model_best.pt
  --stage1-cfg PATH        Stage1 rl_games yaml path. Default: cylinder rotation rl_games_ppo_cfg.yaml
  --device DEVICE          Torch device. Default: cuda:0
  --port PORT              Serial port for the LEAP hand. Default: auto
  --baudrate N             Dynamixel baudrate. Default: 4000000
  --hz N                   Control frequency. Default: 30
  --object-scale X         Cylinder scale used to build the warmup grasp. Default: 1.0
  --warmup-mode MODE       Warmup pose source: auto, analytic, or cache. Default: auto
  --grasp-cache-dir PATH   Optional cached grasp-pose directory for warmup selection
  --grasp-cache-prefix PFX Cached grasp-pose filename prefix. Default: leap_cylinder
  --warmup-seconds N       Seconds spent moving to the initial pose. Default: 4
  --warmup-close-scale X   Extra grasp-closing factor for the warmup pose. Default: 0.15
  --kp N                   Motor P gain. Default: 800
  --kd N                   Motor D gain. Default: 200
  --curr-lim N             Motor current limit in mA. Default: 500
  --read-retries N         Number of joint-state read retries. Default: 8
  --read-retry-interval S  Seconds between joint-state read retries. Default: 0.02
  --print-every N          Print deployment status every N steps. Default: 30
  --max-steps N            Optional deployment step limit. Default: 0 (run until Ctrl+C)
  --dry-run                Skip motor commands and only run inference/polling
  --disable-torque-on-exit Disable torque when the process exits
  --conda-env NAME         Conda env name. Default: env_isaaclab
  --python PYTHON_BIN      Override python executable inside the activated env
  -h, --help               Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/source/LEAP_Isaaclab"
DEFAULT_STAGE1_CFG="${SOURCE_ROOT}/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/agents/rl_games_ppo_cfg.yaml"
STAGE2_LOG_ROOT="${PROJECT_ROOT}/logs/hora_stage2/leap_hand_cylinder_rotation"
PREFERRED_DEPLOY_REFINED_CHECKPOINT="${PROJECT_ROOT}/pretrained/stage2_deploy_refined.pt"

CHECKPOINT=""
USE_LAST_CHECKPOINT=0
STAGE1_CFG="${DEFAULT_STAGE1_CFG}"
DEVICE="cuda:0"
PORT="auto"
BAUDRATE="4000000"
HZ="30"
OBJECT_SCALE="1.0"
WARMUP_MODE="auto"
GRASP_CACHE_DIR=""
GRASP_CACHE_PREFIX="leap_cylinder"
WARMUP_SECONDS="4"
WARMUP_CLOSE_SCALE="0.15"
KP="800"
KD="200"
CURR_LIM="500"
PRINT_EVERY="30"
MAX_STEPS="0"
READ_RETRIES="8"
READ_RETRY_INTERVAL="0.02"
DRY_RUN=0
DISABLE_TORQUE_ON_EXIT=0
CONDA_ENV_NAME="env_isaaclab"
PYTHON_BIN=""
EXTRA_ARGS=()

auto_find_stage2_checkpoint() {
    local pattern="$1"
    find "${STAGE2_LOG_ROOT}" -type f -path "${pattern}" 2>/dev/null | sort | tail -n 1 || true
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --use-last-checkpoint)
            USE_LAST_CHECKPOINT=1
            shift
            ;;
        --stage1-cfg)
            STAGE1_CFG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --baudrate)
            BAUDRATE="$2"
            shift 2
            ;;
        --hz)
            HZ="$2"
            shift 2
            ;;
        --object-scale)
            OBJECT_SCALE="$2"
            shift 2
            ;;
        --warmup-mode)
            WARMUP_MODE="$2"
            shift 2
            ;;
        --grasp-cache-dir)
            GRASP_CACHE_DIR="$2"
            shift 2
            ;;
        --grasp-cache-prefix)
            GRASP_CACHE_PREFIX="$2"
            shift 2
            ;;
        --warmup-seconds)
            WARMUP_SECONDS="$2"
            shift 2
            ;;
        --warmup-close-scale)
            WARMUP_CLOSE_SCALE="$2"
            shift 2
            ;;
        --kp)
            KP="$2"
            shift 2
            ;;
        --kd)
            KD="$2"
            shift 2
            ;;
        --curr-lim)
            CURR_LIM="$2"
            shift 2
            ;;
        --print-every)
            PRINT_EVERY="$2"
            shift 2
            ;;
        --read-retries)
            READ_RETRIES="$2"
            shift 2
            ;;
        --read-retry-interval)
            READ_RETRY_INTERVAL="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --disable-torque-on-exit)
            DISABLE_TORQUE_ON_EXIT=1
            shift
            ;;
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            print_help
            exit 1
            ;;
    esac
done


if [[ ! -f "${STAGE1_CFG}" ]]; then
    echo "[ERROR] Stage1 config not found: ${STAGE1_CFG}" >&2
    exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
    if [[ "${USE_LAST_CHECKPOINT}" -eq 1 ]]; then
        CHECKPOINT="$(auto_find_stage2_checkpoint '*/nn/model_last.pt')"
    elif [[ -f "${PREFERRED_DEPLOY_REFINED_CHECKPOINT}" ]]; then
        CHECKPOINT="${PREFERRED_DEPLOY_REFINED_CHECKPOINT}"
    else
        CHECKPOINT="$(auto_find_stage2_checkpoint '*/nn/model_best.pt')"
    fi
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] Could not auto-detect a stage2 checkpoint under ${STAGE2_LOG_ROOT}" >&2
    echo "        Preferred pretrained checkpoint: ${PREFERRED_DEPLOY_REFINED_CHECKPOINT}" >&2
    echo "        Please pass --checkpoint pretrained/stage2_deploy_refined.pt" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Stage2 checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "/home/tools/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "/home/tools/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Conda is not available in the current shell." >&2
        exit 1
    fi
else
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

echo "[INFO] Activating conda env: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export LEAP_ISAACLAB_SKIP_EXTENSION_IMPORTS=1

CMD=(
    "${PYTHON_BIN}"
    source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts/cylinder_rotation_stage2.py
    --stage2-checkpoint "${CHECKPOINT}"
    --stage1-cfg "${STAGE1_CFG}"
    --device "${DEVICE}"
    --port "${PORT}"
    --baudrate "${BAUDRATE}"
    --hz "${HZ}"
    --object-scale "${OBJECT_SCALE}"
    --warmup-mode "${WARMUP_MODE}"
    --grasp-cache-prefix "${GRASP_CACHE_PREFIX}"
    --warmup-seconds "${WARMUP_SECONDS}"
    --warmup-close-scale "${WARMUP_CLOSE_SCALE}"
    --kp "${KP}"
    --kd "${KD}"
    --curr-lim "${CURR_LIM}"
    --read-retries "${READ_RETRIES}"
    --read-retry-interval "${READ_RETRY_INTERVAL}"
    --print-every "${PRINT_EVERY}"
    --max-steps "${MAX_STEPS}"
)

if [[ -n "${GRASP_CACHE_DIR}" ]]; then
    CMD+=(--grasp-cache-dir "${GRASP_CACHE_DIR}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    CMD+=(--dry-run)
fi

if [[ "${DISABLE_TORQUE_ON_EXIT}" -eq 1 ]]; then
    CMD+=(--disable-torque-on-exit)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Stage2 checkpoint: ${CHECKPOINT}"
echo "[INFO] Stage1 config: ${STAGE1_CFG}"
echo "[INFO] Launch command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
