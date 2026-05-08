#!/usr/bin/env bash

set -euo pipefail

print_help() {
    cat <<'EOF'
Usage: ./scripts/tools/tensorboard_stage2.sh [options]

Options:
  --logdir DIR         Stage2 TensorBoard log root.
                       Default: logs/hora_stage2/leap_hand_cylinder_rotation
  --run-name NAME      Optional specific run name. Default: latest run under logdir
  --host HOST          Bind host. Default: 127.0.0.1
  --port PORT          Bind port. Default: 6007
  --conda-env NAME     Conda env name. Default: env_isaaclab
  --reload N           Reload interval in seconds. Default: 5
  -h, --help           Show this help message

Examples:
  ./scripts/tools/tensorboard_stage2.sh
  ./scripts/tools/tensorboard_stage2.sh --run-name no_objpos_stage2_2026-04-16
  ./scripts/tools/tensorboard_stage2.sh --host 0.0.0.0 --port 6008
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOGDIR="${PROJECT_ROOT}/logs/hora_stage2/leap_hand_cylinder_rotation"
RUN_NAME=""
HOST="127.0.0.1"
PORT="6007"
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

if [[ ! -d "${LOGDIR}" ]]; then
    echo "[ERROR] Stage2 log root does not exist: ${LOGDIR}" >&2
    exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(find "${LOGDIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 | xargs -r basename)"
fi

if [[ -z "${RUN_NAME}" ]]; then
    echo "[ERROR] No stage2 run found under: ${LOGDIR}" >&2
    exit 1
fi

RUN_LOGDIR="${LOGDIR}/${RUN_NAME}"
if [[ ! -d "${RUN_LOGDIR}" ]]; then
    echo "[ERROR] Stage2 run does not exist: ${RUN_LOGDIR}" >&2
    exit 1
fi

echo "[INFO] TensorBoard run: ${RUN_NAME}"
echo "[INFO] TensorBoard logdir: ${RUN_LOGDIR}"
echo "[INFO] Open: http://${HOST}:${PORT}"

exec python -m tensorboard.main \
    --logdir "${RUN_LOGDIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --reload_interval "${RELOAD_INTERVAL}"
