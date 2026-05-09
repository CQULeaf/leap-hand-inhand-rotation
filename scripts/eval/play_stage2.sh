#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./scripts/eval/play_stage2.sh [options] [-- extra isaaclab args]

Options:
  --task TASK              Gym task name. Default: Isaac-CylinderRotation-Leap
  --num-envs N             Number of play environments. Default: 1
  --device DEVICE          Device passed to IsaacLab. Default: cuda:0
  --conda-env NAME         Conda env name. Default: env_isaaclab
  --stage1-cfg PATH        Stage1 rl_games yaml path. Default: cylinder rotation rl_games_ppo_cfg.yaml
  --checkpoint PATH        Explicit stage2 checkpoint path
  --use-last-checkpoint    Load latest stage2 model_last.pt instead of model_best.pt
  --fixed-eval             Disable privileged randomization for cleaner evaluation
  --video                  Record a play video
  --video-length N         Video length. Default: 300
  -h, --help               Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/source/LEAP_Isaaclab"
DEFAULT_STAGE1_CFG="${SOURCE_ROOT}/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/agents/rl_games_ppo_cfg.yaml"
STAGE2_LOG_ROOT="${PROJECT_ROOT}/logs/hora_stage2/leap_hand_cylinder_rotation"
PRETRAINED_STAGE2_CHECKPOINT="${PROJECT_ROOT}/pretrained/stage2_deploy_refined.pt"

TASK="Isaac-CylinderRotation-Leap"
NUM_ENVS="1"
DEVICE="cuda:0"
CONDA_ENV_NAME="env_isaaclab"
STAGE1_CFG="${DEFAULT_STAGE1_CFG}"
VIDEO_LENGTH="300"
CHECKPOINT=""
USE_LAST_CHECKPOINT=0
FIXED_EVAL=0
VIDEO=0
EXTRA_ARGS=()

auto_find_stage2_checkpoint() {
    local pattern="$1"
    find "${STAGE2_LOG_ROOT}" -type f -path "${pattern}" 2>/dev/null | sort | tail -n 1 || true
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --stage1-cfg)
            STAGE1_CFG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --use-last-checkpoint)
            USE_LAST_CHECKPOINT=1
            shift
            ;;
        --fixed-eval)
            FIXED_EVAL=1
            shift
            ;;
        --video)
            VIDEO=1
            shift
            ;;
        --video-length)
            VIDEO_LENGTH="$2"
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
    elif [[ -f "${PRETRAINED_STAGE2_CHECKPOINT}" ]]; then
        CHECKPOINT="${PRETRAINED_STAGE2_CHECKPOINT}"
    else
        CHECKPOINT="$(auto_find_stage2_checkpoint '*/nn/model_best.pt')"
    fi
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] Could not auto-detect a stage2 checkpoint from ${PRETRAINED_STAGE2_CHECKPOINT} or ${STAGE2_LOG_ROOT}" >&2
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

export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

CMD=(
    python
    scripts/internal/hora_play_stage2.py
    --task "${TASK}"
    --stage1-cfg "${STAGE1_CFG}"
    --stage2-checkpoint "${CHECKPOINT}"
    --num-envs "${NUM_ENVS}"
    --device "${DEVICE}"
)

if [[ "${FIXED_EVAL}" -eq 1 ]]; then
    CMD+=(--fixed-eval)
fi

if [[ "${VIDEO}" -eq 1 ]]; then
    CMD+=(--video --video_length "${VIDEO_LENGTH}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Stage1 config: ${STAGE1_CFG}"
echo "[INFO] Stage2 checkpoint: ${CHECKPOINT}"
echo "[INFO] Launch command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
