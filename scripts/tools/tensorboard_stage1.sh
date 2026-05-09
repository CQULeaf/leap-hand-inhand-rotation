#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./scripts/tools/tensorboard_stage1.sh [options]

Options:
  --logdir DIR         TensorBoard log directory.
                       Default: logs/rl_games/leap_hand_cylinder_rotation
  --run-name NAME      Optional specific run name. Default: all runs under logdir
  --host HOST          Bind host. Default: 127.0.0.1
  --port PORT          Bind port. Default: 6006
  --conda-env NAME     Conda env name. Default: env_isaaclab
  --reload N           Reload interval in seconds. Default: 5
  -h, --help           Show this help message

Examples:
  ./scripts/tools/tensorboard_stage1.sh
  ./scripts/tools/tensorboard_stage1.sh --run-name 2026-04-16_15-18-27
  ./scripts/tools/tensorboard_stage1.sh --host 0.0.0.0 --port 6007
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOGDIR="${PROJECT_ROOT}/logs/rl_games/leap_hand_cylinder_rotation"
RUN_NAME=""
HOST="127.0.0.1"
PORT="6006"
CONDA_ENV_NAME="env_isaaclab"
RELOAD_INTERVAL="5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --reload)
            RELOAD_INTERVAL="$2"
            shift 2
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

mkdir -p "${LOGDIR}"

if [[ -n "${RUN_NAME}" ]]; then
    LOGDIR="${LOGDIR}/${RUN_NAME}"
fi

if [[ ! -d "${LOGDIR}" ]]; then
    echo "[ERROR] Stage1 TensorBoard log directory does not exist: ${LOGDIR}" >&2
    exit 1
fi

echo "[INFO] TensorBoard logdir: ${LOGDIR}"
echo "[INFO] Open: http://${HOST}:${PORT}"

exec python -m tensorboard.main \
    --logdir "${LOGDIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --reload_interval "${RELOAD_INTERVAL}"
