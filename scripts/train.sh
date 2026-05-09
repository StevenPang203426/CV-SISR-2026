#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train.sh          # default scale: 2
#   bash train.sh 3        # run srcnn/edsr/espcn on x3
#   bash train.sh 2 4      # run srcnn/edsr/espcn on x2 and x4 (6 windows)

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Please install tmux first." >&2
    exit 1
fi

# 始终切换到项目根目录（scripts/ 的上一级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

MODELS=(srcnn edsr espcn)
SESSION_NAME="${TMUX_SESSION_NAME:-sisr_train}"

if [[ $# -eq 0 ]]; then
    SCALES=(2)
else
    SCALES=("$@")
fi

for s in "${SCALES[@]}"; do
    case "$s" in
        2|3|4) ;;
        *)
            echo "Invalid scale: $s. Allowed values: 2, 3, 4." >&2
            exit 1
            ;;
    esac
done

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists." >&2
    echo "Use: tmux attach -t $SESSION_NAME" >&2
    exit 1
fi

first_scale="${SCALES[0]}"
first_model="${MODELS[0]}"
first_window="${first_model}_x${first_scale}"
first_cmd="python train.py --config configs/${first_model}_x${first_scale}.yaml; rc=\$?; echo \"[${first_window}] exit code: \$rc\"; echo 'Press Enter to close...'; read -r"

tmux new-session -d -s "$SESSION_NAME" -n "$first_window" "$first_cmd"

for s in "${SCALES[@]}"; do
    for m in "${MODELS[@]}"; do
        if [[ "$s" == "$first_scale" && "$m" == "$first_model" ]]; then
            continue
        fi
        window="${m}_x${s}"
        cmd="python train.py --config configs/${m}_x${s}.yaml; rc=\$?; echo \"[${window}] exit code: \$rc\"; echo 'Press Enter to close...'; read -r"
        tmux new-window -t "$SESSION_NAME" -n "$window" "$cmd"
    done
done

echo "Started tmux session: $SESSION_NAME"
echo "Models: ${MODELS[*]}"
echo "Scales: ${SCALES[*]}"

auto_attach="${AUTO_ATTACH:-0}"
if [[ "$auto_attach" == "1" && -z "${TMUX:-}" ]]; then
    tmux attach -t "$SESSION_NAME"
else
    echo "Attach manually: tmux attach -t $SESSION_NAME"
fi
