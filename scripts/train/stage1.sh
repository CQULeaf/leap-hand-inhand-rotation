#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./scripts/train/stage1.sh [options] [-- extra isaaclab args]

Options:
  --profile NAME            Training preset: local-5060, cloud-4090, debug
                            Default: local-5060
  --task TASK                Gym task name. Default: Isaac-CylinderRotation-Leap
  --num-envs N               Number of parallel environments. Default depends on profile
  --max-iterations N         PPO max iterations. Default depends on profile
  --seed N                   Random seed. Default: 42
  --device DEVICE            Device passed to IsaacLab. Default: cuda:0
  --checkpoint PATH          Resume training from an existing rl_games checkpoint
  --resume-from PATH         Alias of --checkpoint
  --run-name NAME            Optional run directory name
  --conda-env NAME           Conda env name. Default: env_isaaclab
  --python PYTHON_BIN        Override python executable inside the activated env
  --omp-threads N            Set OMP_NUM_THREADS/MKL_NUM_THREADS. Default depends on profile
  --headless / --no-headless Enable or disable headless mode. Default: headless
  -h, --help                 Show this help message

Examples:
  ./scripts/train/stage1.sh
  ./scripts/train/stage1.sh --profile local-5060
  ./scripts/train/stage1.sh --profile cloud-4090
  ./scripts/train/stage1.sh --checkpoint /abs/path/to/ep200.pth --max-iterations 400
  ./scripts/train/stage1.sh --seed 7 -- --video --video_length 300
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/source/LEAP_Isaaclab"

TASK="Isaac-CylinderRotation-Leap"
PROFILE="local-5060"
NUM_ENVS=""
MAX_ITERATIONS=""
SEED="42"
DEVICE="cuda:0"
CHECKPOINT=""
RUN_NAME=""
CONDA_ENV_NAME="env_isaaclab"
HEADLESS=1
PYTHON_BIN=""
OMP_THREADS=""
EXTRA_ARGS=()
RLG_HORIZON_LENGTH="96"
RLG_MINIBATCH_SIZE="3072"
RLG_SEQ_LENGTH="12"

apply_profile_defaults() {
    case "${PROFILE}" in
        local-5060)
            NUM_ENVS="${NUM_ENVS:-3072}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-1800}"
            OMP_THREADS="${OMP_THREADS:-12}"
            ;;
        cloud-4090)
            NUM_ENVS="${NUM_ENVS:-4096}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-2500}"
            OMP_THREADS="${OMP_THREADS:-12}"
            ;;
        debug)
            NUM_ENVS="${NUM_ENVS:-256}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
            OMP_THREADS="${OMP_THREADS:-4}"
            ;;
        *)
            echo "[ERROR] Unknown profile: ${PROFILE}" >&2
            exit 1
            ;;
    esac
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
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
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
        --checkpoint|--resume-from)
            CHECKPOINT="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
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

TOTAL_BATCH_SIZE=$(( NUM_ENVS * RLG_HORIZON_LENGTH ))
if (( TOTAL_BATCH_SIZE % RLG_MINIBATCH_SIZE != 0 )); then
    echo "[ERROR] Incompatible rl_games batch configuration." >&2
    echo "        num_envs * horizon_length = ${TOTAL_BATCH_SIZE}" >&2
    echo "        minibatch_size = ${RLG_MINIBATCH_SIZE}" >&2
    echo "        Expected total batch size to be divisible by minibatch size." >&2
    exit 1
fi
if (( RLG_MINIBATCH_SIZE % RLG_SEQ_LENGTH != 0 )); then
    echo "[ERROR] Incompatible rl_games recurrent configuration." >&2
    echo "        minibatch_size = ${RLG_MINIBATCH_SIZE}" >&2
    echo "        seq_length = ${RLG_SEQ_LENGTH}" >&2
    echo "        Expected minibatch size to be divisible by seq length." >&2
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
    scripts/internal/rl_games_train.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --max_iterations "${MAX_ITERATIONS}"
    --seed "${SEED}"
    --device "${DEVICE}"
)

if [[ -n "${CHECKPOINT}" ]]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi

if [[ "${HEADLESS}" -eq 1 ]]; then
    CMD+=(--headless)
fi

if [[ -n "${RUN_NAME}" ]]; then
    CMD+=(--run_name "${RUN_NAME}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Profile: ${PROFILE}"
echo "[INFO] num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}, omp_threads=${OMP_THREADS}"
if [[ -n "${CHECKPOINT}" ]]; then
    echo "[INFO] Resuming from checkpoint: ${CHECKPOINT}"
    echo "[INFO] Note: --max-iterations is the total target epoch count, not the additional epochs to train."
fi
echo "[INFO] PYTHONPATH prepended with: ${SOURCE_ROOT}"
echo "[INFO] Launch command:"
printf '  %q' "${CMD[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
