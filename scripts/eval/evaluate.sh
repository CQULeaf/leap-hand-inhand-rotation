#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./scripts/eval/evaluate.sh [options] [-- extra isaaclab args]

Options:
  --policy-type TYPE        stage1 or stage2. Required.
  --eval-preset PRESET      fixed, id, or ood. Default: id
  --task TASK               Gym task name. Default: Isaac-CylinderRotation-Leap
  --num-envs N              Number of parallel eval environments. Default: 256
  --num-episodes N          Number of evaluation episodes. Default: 256
  --stage1-checkpoint PATH  Explicit stage1 checkpoint path
  --stage2-checkpoint PATH  Explicit stage2 checkpoint path
  --stage1-cfg PATH         Stage1 rl_games yaml path
  --adapt-encoder-type TYPE Stage2 encoder: auto, tconv, flatten_mlp, or gru. Default: auto
  --adapt-hist-len N        Override adaptation history length
  --latent-dim N            Override stage2 latent dimension
  --seed N                  Passed through to IsaacLab. Default: 42
  --device DEVICE           Device passed to IsaacLab. Default: cuda:0
  --output-dir PATH         Optional result directory
  --run-name NAME           Optional run name
  --fixed-object-scale V    Fixed object scale override
  --fixed-object-mass V     Fixed object mass override in kg
  --fixed-object-friction V Fixed object friction override
  --fixed-object-com V      Fixed CoM offset applied identically to x/y/z
  --conda-env NAME          Conda env name. Default: env_isaaclab
  --headless / --no-headless Enable or disable headless mode. Default: headless
  -h, --help                Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/source/LEAP_Isaaclab"
DEFAULT_STAGE1_CFG="${SOURCE_ROOT}/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/agents/rl_games_ppo_cfg.yaml"
STAGE1_LOG_ROOT="${PROJECT_ROOT}/logs/rl_games/leap_hand_cylinder_rotation"
STAGE2_LOG_ROOT="${PROJECT_ROOT}/logs/hora_stage2/leap_hand_cylinder_rotation"

POLICY_TYPE=""
EVAL_PRESET="id"
TASK="Isaac-CylinderRotation-Leap"
NUM_ENVS="256"
NUM_EPISODES="256"
STAGE1_CHECKPOINT=""
STAGE2_CHECKPOINT=""
STAGE1_CFG="${DEFAULT_STAGE1_CFG}"
ADAPT_ENCODER_TYPE="auto"
ADAPT_HIST_LEN=""
LATENT_DIM=""
SEED="42"
DEVICE="cuda:0"
OUTPUT_DIR=""
RUN_NAME=""
FIXED_OBJECT_SCALE=""
FIXED_OBJECT_MASS=""
FIXED_OBJECT_FRICTION=""
FIXED_OBJECT_COM=""
CONDA_ENV_NAME="env_isaaclab"
HEADLESS=1
EXTRA_ARGS=()

auto_find_stage1_checkpoint() {
    local latest_best=""
    local latest_last=""
    latest_best="$(find "${STAGE1_LOG_ROOT}" -type f -path '*/nn/leap_hand_cylinder_rotation.pth' 2>/dev/null | sort | tail -n 1 || true)"
    latest_last="$(find "${STAGE1_LOG_ROOT}" -type f -path '*/nn/last_*.pth' 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${latest_best}" ]]; then
        printf '%s\n' "${latest_best}"
        return 0
    fi
    if [[ -n "${latest_last}" ]]; then
        printf '%s\n' "${latest_last}"
        return 0
    fi
    return 1
}

auto_find_stage2_checkpoint() {
    local latest_best=""
    local latest_last=""
    latest_best="$(find "${STAGE2_LOG_ROOT}" -type f -path '*/nn/model_best.pt' 2>/dev/null | sort | tail -n 1 || true)"
    latest_last="$(find "${STAGE2_LOG_ROOT}" -type f -path '*/nn/model_last.pt' 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${latest_best}" ]]; then
        printf '%s\n' "${latest_best}"
        return 0
    fi
    if [[ -n "${latest_last}" ]]; then
        printf '%s\n' "${latest_last}"
        return 0
    fi
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-type)
            POLICY_TYPE="$2"
            shift 2
            ;;
        --eval-preset)
            EVAL_PRESET="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --stage1-checkpoint)
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        --stage2-checkpoint)
            STAGE2_CHECKPOINT="$2"
            shift 2
            ;;
        --stage1-cfg)
            STAGE1_CFG="$2"
            shift 2
            ;;
        --adapt-encoder-type)
            ADAPT_ENCODER_TYPE="$2"
            shift 2
            ;;
        --adapt-hist-len)
            ADAPT_HIST_LEN="$2"
            shift 2
            ;;
        --latent-dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --fixed-object-scale)
            FIXED_OBJECT_SCALE="$2"
            shift 2
            ;;
        --fixed-object-mass)
            FIXED_OBJECT_MASS="$2"
            shift 2
            ;;
        --fixed-object-friction)
            FIXED_OBJECT_FRICTION="$2"
            shift 2
            ;;
        --fixed-object-com)
            FIXED_OBJECT_COM="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=1
            shift
            ;;
        --no-headless)
            HEADLESS=0
            shift
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

if [[ -z "${POLICY_TYPE}" ]]; then
    echo "[ERROR] --policy-type is required." >&2
    exit 1
fi

if [[ "${POLICY_TYPE}" != "stage1" && "${POLICY_TYPE}" != "stage2" ]]; then
    echo "[ERROR] --policy-type must be stage1 or stage2." >&2
    exit 1
fi

if [[ ! -f "${STAGE1_CFG}" ]]; then
    echo "[ERROR] Stage1 config not found: ${STAGE1_CFG}" >&2
    exit 1
fi

if [[ "${POLICY_TYPE}" == "stage1" && -z "${STAGE1_CHECKPOINT}" ]]; then
    if ! STAGE1_CHECKPOINT="$(auto_find_stage1_checkpoint)"; then
        echo "[ERROR] Could not auto-detect a stage1 checkpoint under ${STAGE1_LOG_ROOT}" >&2
        exit 1
    fi
fi

if [[ "${POLICY_TYPE}" == "stage2" && -z "${STAGE2_CHECKPOINT}" ]]; then
    if ! STAGE2_CHECKPOINT="$(auto_find_stage2_checkpoint)"; then
        echo "[ERROR] Could not auto-detect a stage2 checkpoint under ${STAGE2_LOG_ROOT}" >&2
        exit 1
    fi
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
    scripts/eval/evaluate_policy.py
    --task "${TASK}"
    --policy-type "${POLICY_TYPE}"
    --stage1-cfg "${STAGE1_CFG}"
    --adapt-encoder-type "${ADAPT_ENCODER_TYPE}"
    --num-envs "${NUM_ENVS}"
    --num-episodes "${NUM_EPISODES}"
    --eval-preset "${EVAL_PRESET}"
    --seed "${SEED}"
    --device "${DEVICE}"
)

if [[ -n "${STAGE1_CHECKPOINT}" ]]; then
    CMD+=(--stage1-checkpoint "${STAGE1_CHECKPOINT}")
fi

if [[ -n "${STAGE2_CHECKPOINT}" ]]; then
    CMD+=(--stage2-checkpoint "${STAGE2_CHECKPOINT}")
fi

if [[ -n "${ADAPT_HIST_LEN}" ]]; then
    CMD+=(--adapt-hist-len "${ADAPT_HIST_LEN}")
fi

if [[ -n "${LATENT_DIM}" ]]; then
    CMD+=(--latent-dim "${LATENT_DIM}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
    CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${RUN_NAME}" ]]; then
    CMD+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${FIXED_OBJECT_SCALE}" ]]; then
    CMD+=(--fixed-object-scale "${FIXED_OBJECT_SCALE}")
fi

if [[ -n "${FIXED_OBJECT_MASS}" ]]; then
    CMD+=(--fixed-object-mass "${FIXED_OBJECT_MASS}")
fi

if [[ -n "${FIXED_OBJECT_FRICTION}" ]]; then
    CMD+=(--fixed-object-friction "${FIXED_OBJECT_FRICTION}")
fi

if [[ -n "${FIXED_OBJECT_COM}" ]]; then
    CMD+=(--fixed-object-com "${FIXED_OBJECT_COM}")
fi

if [[ "${HEADLESS}" -eq 1 ]]; then
    CMD+=(--headless)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Policy type: ${POLICY_TYPE}"
echo "[INFO] Eval preset: ${EVAL_PRESET}"
echo "[INFO] num_envs=${NUM_ENVS}, num_episodes=${NUM_EPISODES}"
echo "[INFO] Stage1 config: ${STAGE1_CFG}"
if [[ -n "${STAGE1_CHECKPOINT}" ]]; then
    echo "[INFO] Stage1 checkpoint: ${STAGE1_CHECKPOINT}"
fi
if [[ -n "${STAGE2_CHECKPOINT}" ]]; then
    echo "[INFO] Stage2 checkpoint: ${STAGE2_CHECKPOINT}"
fi
if [[ -n "${FIXED_OBJECT_SCALE}" ]]; then
    echo "[INFO] Fixed object scale: ${FIXED_OBJECT_SCALE}"
fi
if [[ -n "${FIXED_OBJECT_MASS}" ]]; then
    echo "[INFO] Fixed object mass: ${FIXED_OBJECT_MASS}"
fi
if [[ -n "${FIXED_OBJECT_FRICTION}" ]]; then
    echo "[INFO] Fixed object friction: ${FIXED_OBJECT_FRICTION}"
fi
if [[ -n "${FIXED_OBJECT_COM}" ]]; then
    echo "[INFO] Fixed object com: ${FIXED_OBJECT_COM}"
fi
echo "[INFO] Launch command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
