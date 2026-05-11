#!/usr/bin/env bash
# ===========================================================
# 全模型 × 全规模 × 全数据集 批量测试脚本
# ===========================================================
#
# 用法:
#   bash scripts/test.sh                    # 测试全部（5模型×3规模×2数据集）
#   bash scripts/test.sh --models edsr imdn # 只测 edsr 和 imdn
#   bash scripts/test.sh --scales 2 4       # 只测 x2 和 x4
<<<<<<< HEAD
#   bash scripts/test.sh --test_dir data/datasets/Set5  # 自定义测试目录
=======
#   bash scripts/test.sh --test_dir demo/original  # 自定义测试目录
>>>>>>> 5a1768bb047aa059baf4a7ccca742c8fec77fb5d
#   bash scripts/test.sh --save_images      # 同时保存 SR/LR 图像
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- 默认值 ----
ALL_MODELS=(srcnn espcn edsr)
ALL_SCALES=(2 3 4)
TEST_DIR="demo/original"
SAVE_IMAGES=false

# ---- 参数解析 ----
MODELS=()
SCALES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1"); shift
            done
            ;;
        --scales)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SCALES+=("$1"); shift
            done
            ;;
        --test_dir)
            shift; TEST_DIR="$1"; shift
            ;;
        --save_images)
            SAVE_IMAGES=true; shift
            ;;
        *)
            echo "Unknown option: $1" >&2; exit 1
            ;;
    esac
done

# 使用默认值（如果未指定）
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${ALL_MODELS[@]}")
[[ ${#SCALES[@]} -eq 0 ]] && SCALES=("${ALL_SCALES[@]}")
<<<<<<< HEAD

DATASET_NAME=$(basename "$(realpath "$TEST_DIR")")
=======
>>>>>>> 5a1768bb047aa059baf4a7ccca742c8fec77fb5d

DATASET_NAME=$(basename "$(realpath "$TEST_DIR")")

# ---- 检查点路径查找（experiments/ ） ----
find_ckpt() {
    local model=$1 scale=$2
    local candidates=(
        "experiments/${model}_x${scale}/best.pt"
    )
    for c in "${candidates[@]}"; do
        if [[ -f "$c" ]]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

# ---- 打印计划 ----
total=$(( ${#MODELS[@]} * ${#SCALES[@]} ))
echo "============================================"
echo "  SISR 批量测试"
echo "============================================"
echo "  模型:     ${MODELS[*]}"
echo "  规模:     ${SCALES[*]}"
echo "  测试目录: ${TEST_DIR} (${DATASET_NAME})"
echo "  保存图像: ${SAVE_IMAGES}"
echo "  总计: ${total} 组实验"
echo "============================================"
echo ""

# ---- 执行测试 ----
passed=0
skipped=0
failed=0

for model in "${MODELS[@]}"; do
    for scale in "${SCALES[@]}"; do
        ckpt=$(find_ckpt "$model" "$scale" 2>/dev/null || true)
        if [[ -z "$ckpt" ]]; then
            echo "[SKIP] ${model}_x${scale} — 未找到检查点"
            skipped=$((skipped + 1))
            continue
        fi

        out_dir="experiments/${model}_x${scale}/test/${DATASET_NAME}"
        echo "--------------------------------------------"
        echo "[RUN]  ${model}_x${scale} on ${DATASET_NAME}"
        echo "       ckpt: ${ckpt}"
        echo "       out:  ${out_dir}"

        cmd=(python test.py
            --ckpt "$ckpt"
            --test_dir "$TEST_DIR"
            --model "$model"
            --scale "$scale"
            --out_dir "$out_dir"
        )
        if $SAVE_IMAGES; then
            cmd+=(--save_images)
        fi

        if "${cmd[@]}"; then
            echo "[DONE] ${model}_x${scale} on ${DATASET_NAME}"
            passed=$((passed + 1))
        else
            echo "[FAIL] ${model}_x${scale} on ${DATASET_NAME}"
            failed=$((failed + 1))
        fi
        echo ""
    done
done

# ---- 汇总 ----
echo "============================================"
echo "  测试完成"
echo "  通过: ${passed}  跳过: ${skipped}  失败: ${failed}"
echo "============================================"

if [[ $failed -gt 0 ]]; then
    exit 1
fi
