#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./scripts/train/stage2.sh [options] [-- extra isaaclab args]

Options:
  --profile NAME             Training preset: local-5060, cloud-4090, debug
                             Default: local-5060
  --task TASK                Gym task name. Default: Isaac-CylinderRotation-Leap
  --stage1-checkpoint PATH   Stage1 checkpoint path. Default: auto-detect latest best stage1 checkpoint
  --stage2-checkpoint PATH   Optional stage2 checkpoint path for fine-tuning / resume
  --stage1-cfg PATH          Stage1 rl_games yaml path. Default: cylinder rotation rl_games_ppo_cfg.yaml
  --num-envs N               Number of parallel environments. Default depends on profile
  --max-steps N              Total stage2 environment steps. Default depends on profile
  --learning-rate LR         Adaptation optimizer learning rate. Default: 3e-4
  --action-loss-weight W     Weight for teacher-student action imitation loss. Default: 0.0
  --adapt-encoder-type TYPE  Stage2 encoder: tconv, flatten_mlp, or gru. Default: tconv
  --adapt-hist-len N         Override adaptation history length, e.g. 25/30/40
  --latent-dim N             Override latent dimension, e.g. 9 for direct privileged input
  --save-every N             Stage2 checkpoint save interval. Default: 500000
  --log-every N              TensorBoard / console log interval. Default: 5000
  --run-name NAME            Optional run name. Default: timestamp
  --seed N                   Passed through to IsaacLab. Default: 42
  --device DEVICE            Device passed to IsaacLab. Default: cuda:0
  --conda-env NAME           Conda env name. Default: env_isaaclab
  --python PYTHON_BIN        Override python executable inside the activated env
  --omp-threads N            Set OMP_NUM_THREADS/MKL_NUM_THREADS. Default depends on profile
  --headless / --no-headless Enable or disable headless mode. Default: headless
  -h, --help                 Show this help message

Examples:
  ./scripts/train/stage2.sh
  ./scripts/train/stage2.sh --profile debug
  ./scripts/train/stage2.sh --stage1-checkpoint /abs/path/to/stage1.pth
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/source/LEAP_Isaaclab"
DEFAULT_STAGE1_CFG="${SOURCE_ROOT}/LEAP_Isaaclab/tasks/leap_hand_cylinder_rotation/agents/rl_games_ppo_cfg.yaml"
STAGE1_LOG_ROOT="${PROJECT_ROOT}/logs/rl_games/leap_hand_cylinder_rotation"

TASK="Isaac-CylinderRotation-Leap"
PROFILE="local-5060"
STAGE1_CHECKPOINT=""
STAGE2_CHECKPOINT=""
STAGE1_CFG="${DEFAULT_STAGE1_CFG}"
NUM_ENVS=""
MAX_STEPS=""
LEARNING_RATE="3e-4"
ACTION_LOSS_WEIGHT="0.0"
ADAPT_ENCODER_TYPE="tconv"
ADAPT_HIST_LEN=""
LATENT_DIM=""
SAVE_EVERY="500000"
LOG_EVERY="5000"
RUN_NAME=""
SEED="42"
DEVICE="cuda:0"
CONDA_ENV_NAME="env_isaaclab"
HEADLESS=1
PYTHON_BIN=""
OMP_THREADS=""
EXTRA_ARGS=()

apply_profile_defaults() {
    case "${PROFILE}" in
        local-5060)
            NUM_ENVS="${NUM_ENVS:-3072}"
            MAX_STEPS="${MAX_STEPS:-5000000}"
            OMP_THREADS="${OMP_THREADS:-12}"
            ;;
        cloud-4090)
            NUM_ENVS="${NUM_ENVS:-4096}"
            MAX_STEPS="${MAX_STEPS:-8000000}"
            OMP_THREADS="${OMP_THREADS:-12}"
            ;;
        debug)
            NUM_ENVS="${NUM_ENVS:-256}"
            MAX_STEPS="${MAX_STEPS:-50000}"
            OMP_THREADS="${OMP_THREADS:-4}"
            ;;
        *)
            echo "[ERROR] Unknown profile: ${PROFILE}" >&2
            exit 1
            ;;
    esac
}

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
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
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --action-loss-weight)
            ACTION_LOSS_WEIGHT="$2"
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
        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --log-every)
            LOG_EVERY="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
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
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --omp-threads)
            OMP_THREADS="$2"
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

apply_profile_defaults

if [[ ! -d "${SOURCE_ROOT}" ]]; then
    echo "[ERROR] LEAP_Isaaclab source directory not found: ${SOURCE_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${STAGE1_CFG}" ]]; then
    echo "[ERROR] Stage1 config not found: ${STAGE1_CFG}" >&2
    exit 1
fi

if [[ -z "${STAGE1_CHECKPOINT}" ]]; then
    if ! STAGE1_CHECKPOINT="$(auto_find_stage1_checkpoint)"; then
        echo "[ERROR] Could not auto-detect a stage1 checkpoint under ${STAGE1_LOG_ROOT}" >&2
        echo "        Please pass --stage1-checkpoint /abs/path/to/stage1.pth" >&2
        exit 1
    fi
fi

if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "[ERROR] Stage1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
    exit 1
fi

if [[ -n "${STAGE2_CHECKPOINT}" && ! -f "${STAGE2_CHECKPOINT}" ]]; then
    echo "[ERROR] Stage2 checkpoint not found: ${STAGE2_CHECKPOINT}" >&2
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
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplconfig"
export OMP_NUM_THREADS="${OMP_THREADS}"
export MKL_NUM_THREADS="${OMP_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_THREADS}"
mkdir -p "${MPLCONFIGDIR}"

CMD=(
    "${PYTHON_BIN}"
    scripts/internal/hora_train_stage2.py
    --task "${TASK}"
    --stage1-cfg "${STAGE1_CFG}"
    --stage1-checkpoint "${STAGE1_CHECKPOINT}"
    --num-envs "${NUM_ENVS}"
    --max-steps "${MAX_STEPS}"
    --learning-rate "${LEARNING_RATE}"
    --action-loss-weight "${ACTION_LOSS_WEIGHT}"
    --adapt-encoder-type "${ADAPT_ENCODER_TYPE}"
    --save-every "${SAVE_EVERY}"
    --log-every "${LOG_EVERY}"
    --seed "${SEED}"
    --device "${DEVICE}"
)

if [[ -n "${RUN_NAME}" ]]; then
    CMD+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${ADAPT_HIST_LEN}" ]]; then
    CMD+=(--adapt-hist-len "${ADAPT_HIST_LEN}")
fi

if [[ -n "${LATENT_DIM}" ]]; then
    CMD+=(--latent-dim "${LATENT_DIM}")
fi

if [[ -n "${STAGE2_CHECKPOINT}" ]]; then
    CMD+=(--stage2-checkpoint "${STAGE2_CHECKPOINT}")
fi

if [[ "${HEADLESS}" -eq 1 ]]; then
    CMD+=(--headless)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Profile: ${PROFILE}"
echo "[INFO] num_envs=${NUM_ENVS}, max_steps=${MAX_STEPS}, omp_threads=${OMP_THREADS}"
echo "[INFO] Stage1 config: ${STAGE1_CFG}"
echo "[INFO] Stage1 checkpoint: ${STAGE1_CHECKPOINT}"
echo "[INFO] PYTHONPATH prepended with: ${SOURCE_ROOT}"
echo "[INFO] Launch command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
