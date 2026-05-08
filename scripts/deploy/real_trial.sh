#!/usr/bin/env bash
set -u

print_help() {
    cat <<'HELP'
Usage: deploy_real_trial.sh [OPTIONS]

Run one real-world Stage2 deployment trial and append metadata to a CSV table.

Object metadata:
  --object-id ID              Object id. Default: object01
  --object-name NAME          Object name. Default: id_cylinder
  --object-category CAT       ID or OOD. Default: ID
  --object-shape SHAPE        Object shape. Default: cylinder
  --diameter-cm X             Object diameter in cm. Default: 7.5
  --height-cm X               Object height in cm. Default: 7.5
  --material NAME             Object material. Default: PLA
  --mass-g X                  Object mass in grams. Default: unknown
  --com-note TEXT             COM description. Default: unknown
  --object-note TEXT          Object notes. Default: 3D printed PLA cylinder

Trial outcome metadata, can be filled before or edited after the trial:
  --success VALUE             yes/no/partial/pending. Default: pending
  --ttf-s X                   Time-to-fall seconds or observed duration. Default: pending
  --net-turns-est X           Estimated net turns. Default: pending
  --fall-or-stop VALUE        timeout/fall/manual_stop/pending. Default: pending
  --video-file PATH           Associated video path. Default: empty
  --notes TEXT                Notes. Default: empty

Deployment parameters:
  --checkpoint PATH           Stage2 checkpoint path. Default: Deploy-Refined model_best.pt
  --port PATH                 Serial port. Default: /dev/ttyUSB0
  --hz N                      Control frequency. Default: 30
  --warmup-mode MODE          Warmup mode. Default: analytic
  --warmup-seconds N          Warmup seconds. Default: 5
  --warmup-close-scale X      Warmup close scale. Default: 0.10
  --kp N                      P gain. Default: 600
  --kd N                      D gain. Default: 150
  --curr-lim N                Current limit mA. Default: 400
  --max-steps N               Max deployment steps. Default: 600
  --table PATH                CSV output table. Default: evaluation/real/tables/real_deploy_trials_YYYYMMDD.csv
  --trial-id ID               Trial id. Default: auto YYYYMMDD_objectXX_trialNN
  -h, --help                  Show this help message
HELP
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATE_STR="$(date +%Y%m%d)"
DATE_ISO="$(date +%F)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "[ERROR] No python interpreter found for CSV bookkeeping. Set PYTHON_BIN explicitly if needed." >&2
        exit 1
    fi
fi

CHECKPOINT="logs/hora_stage2/leap_hand_cylinder_rotation/stage2_orig_teacher_comwide_actloss1_2026-04-16/nn/model_best.pt"
PORT="/dev/ttyUSB0"
HZ="30"
WARMUP_MODE="analytic"
WARMUP_SECONDS="5"
WARMUP_CLOSE_SCALE="0.10"
KP="600"
KD="150"
CURR_LIM="400"
MAX_STEPS="600"
TABLE="evaluation/real/tables/real_deploy_trials_${DATE_STR}.csv"
TRIAL_ID=""

OBJECT_ID="object01"
OBJECT_NAME="id_cylinder"
OBJECT_CATEGORY="ID"
OBJECT_SHAPE="cylinder"
DIAMETER_CM="7.5"
HEIGHT_CM="7.5"
MATERIAL="PLA"
MASS_G="unknown"
COM_NOTE="unknown"
OBJECT_NOTE="3D printed PLA cylinder"

SUCCESS="pending"
TTF_S="pending"
NET_TURNS_EST="pending"
FALL_OR_STOP="pending"
VIDEO_FILE=""
NOTES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --object-id) OBJECT_ID="$2"; shift 2 ;;
        --object-name) OBJECT_NAME="$2"; shift 2 ;;
        --object-category) OBJECT_CATEGORY="$2"; shift 2 ;;
        --object-shape) OBJECT_SHAPE="$2"; shift 2 ;;
        --diameter-cm) DIAMETER_CM="$2"; shift 2 ;;
        --height-cm) HEIGHT_CM="$2"; shift 2 ;;
        --material) MATERIAL="$2"; shift 2 ;;
        --mass-g) MASS_G="$2"; shift 2 ;;
        --com-note) COM_NOTE="$2"; shift 2 ;;
        --object-note) OBJECT_NOTE="$2"; shift 2 ;;
        --success) SUCCESS="$2"; shift 2 ;;
        --ttf-s) TTF_S="$2"; shift 2 ;;
        --net-turns-est) NET_TURNS_EST="$2"; shift 2 ;;
        --fall-or-stop) FALL_OR_STOP="$2"; shift 2 ;;
        --video-file) VIDEO_FILE="$2"; shift 2 ;;
        --notes) NOTES="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --hz) HZ="$2"; shift 2 ;;
        --warmup-mode) WARMUP_MODE="$2"; shift 2 ;;
        --warmup-seconds) WARMUP_SECONDS="$2"; shift 2 ;;
        --warmup-close-scale) WARMUP_CLOSE_SCALE="$2"; shift 2 ;;
        --kp) KP="$2"; shift 2 ;;
        --kd) KD="$2"; shift 2 ;;
        --curr-lim) CURR_LIM="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --table) TABLE="$2"; shift 2 ;;
        --trial-id) TRIAL_ID="$2"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; print_help; exit 1 ;;
    esac
done

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${TABLE}")" evaluation/real/logs

if [[ ! -f "${TABLE}" ]]; then
    cat > "${TABLE}" <<'HEADER'
trial_id,date,start_time,end_time,object_id,object_name,object_category,object_shape,object_diameter_cm,object_height_cm,material,mass_g,com_note,object_note,checkpoint,hz,kp,kd,curr_lim,warmup_mode,warmup_seconds,warmup_close_scale,max_steps,exit_code,terminal_status,success,ttf_s,net_turns_est,fall_or_stop,video_file,log_file,notes
HEADER
fi

if [[ -z "${TRIAL_ID}" ]]; then
    TRIAL_ID="$(${PYTHON_BIN} - "${TABLE}" "${DATE_STR}" "${OBJECT_ID}" <<'PY'
import csv
import os
import re
import sys

table, date_str, object_id = sys.argv[1:4]
max_idx = 0
if os.path.exists(table):
    with open(table, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tid = row.get('trial_id', '')
            m = re.fullmatch(rf'{re.escape(date_str)}_{re.escape(object_id)}_trial(\d+)', tid)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
print(f'{date_str}_{object_id}_trial{max_idx + 1:02d}')
PY
)"
fi

LOG_FILE="evaluation/real/logs/${TRIAL_ID}.log"
START_TIME="$(date --iso-8601=seconds)"
START_EPOCH="$(date +%s)"

set +e
./scripts/deploy/stage2.sh \
    --checkpoint "${CHECKPOINT}" \
    --port "${PORT}" \
    --hz "${HZ}" \
    --warmup-mode "${WARMUP_MODE}" \
    --warmup-seconds "${WARMUP_SECONDS}" \
    --warmup-close-scale "${WARMUP_CLOSE_SCALE}" \
    --kp "${KP}" \
    --kd "${KD}" \
    --curr-lim "${CURR_LIM}" \
    --max-steps "${MAX_STEPS}" \
    --disable-torque-on-exit 2>&1 | tee "${LOG_FILE}"
PIPE_STATUS=(${PIPESTATUS[@]})
EXIT_CODE="${PIPE_STATUS[0]}"
set -e

END_TIME="$(date --iso-8601=seconds)"
END_EPOCH="$(date +%s)"
if [[ "${TTF_S}" == "pending" && "${EXIT_CODE}" == "0" ]]; then
    TTF_S="$(${PYTHON_BIN} - "${MAX_STEPS}" "${HZ}" <<'PY'
import sys
max_steps = float(sys.argv[1])
hz = float(sys.argv[2])
duration = max_steps / hz if hz else 0.0
print(f'{duration:g}')
PY
)"
fi
if [[ "${FALL_OR_STOP}" == "pending" && "${EXIT_CODE}" == "0" ]]; then
    FALL_OR_STOP="timeout"
fi
if [[ "${EXIT_CODE}" == "0" ]]; then
    TERMINAL_STATUS="completed_max_steps_or_clean_exit"
else
    TERMINAL_STATUS="process_failed"
fi

"${PYTHON_BIN}" - "${TABLE}" \
    "${TRIAL_ID}" "${DATE_ISO}" "${START_TIME}" "${END_TIME}" \
    "${OBJECT_ID}" "${OBJECT_NAME}" "${OBJECT_CATEGORY}" "${OBJECT_SHAPE}" \
    "${DIAMETER_CM}" "${HEIGHT_CM}" "${MATERIAL}" "${MASS_G}" "${COM_NOTE}" "${OBJECT_NOTE}" \
    "${CHECKPOINT}" "${HZ}" "${KP}" "${KD}" "${CURR_LIM}" "${WARMUP_MODE}" "${WARMUP_SECONDS}" "${WARMUP_CLOSE_SCALE}" "${MAX_STEPS}" \
    "${EXIT_CODE}" "${TERMINAL_STATUS}" "${SUCCESS}" "${TTF_S}" "${NET_TURNS_EST}" "${FALL_OR_STOP}" "${VIDEO_FILE}" "${LOG_FILE}" "${NOTES}" <<'PY'
import csv
import sys

path = sys.argv[1]
fields = [
    'trial_id','date','start_time','end_time','object_id','object_name','object_category','object_shape',
    'object_diameter_cm','object_height_cm','material','mass_g','com_note','object_note','checkpoint','hz',
    'kp','kd','curr_lim','warmup_mode','warmup_seconds','warmup_close_scale','max_steps','exit_code',
    'terminal_status','success','ttf_s','net_turns_est','fall_or_stop','video_file','log_file','notes'
]
row = dict(zip(fields, sys.argv[2:]))
with open(path, 'a', newline='', encoding='utf-8') as f:
    csv.DictWriter(f, fieldnames=fields, lineterminator='\n').writerow(row)
print(f'[INFO] Trial appended: {path} :: {row["trial_id"]}')
PY

exit "${EXIT_CODE}"
