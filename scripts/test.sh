#!/usr/bin/env bash
# ===========================================================
# 全模型 × 全规模 × 全数据集 批量测试脚本
# ===========================================================
#
# 用法:
#   bash scripts/test.sh                    # 测试全部（5模型×3规模×2数据集）
#   bash scripts/test.sh --models edsr imdn # 只测 edsr 和 imdn
#   bash scripts/test.sh --scales 2 4       # 只测 x2 和 x4
#   bash scripts/test.sh --datasets Set5    # 只在 Set5 上测
#   bash scripts/test.sh --save_images      # 同时保存 SR/LR 图像
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- 默认值 ----
ALL_MODELS=(srcnn fsrcnn espcn edsr imdn)
ALL_SCALES=(2 3 4)
ALL_DATASETS=(Set5 Set14)
SAVE_IMAGES=false

# ---- 参数解析 ----
MODELS=()
SCALES=()
DATASETS=()

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
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1"); shift
            done
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
[[ ${#MODELS[@]} -eq 0 ]]   && MODELS=("${ALL_MODELS[@]}")
[[ ${#SCALES[@]} -eq 0 ]]   && SCALES=("${ALL_SCALES[@]}")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DATASETS[@]}")

# ---- 检查点路径查找（兼容 experiments/ 和 output/ 两种目录） ----
find_ckpt() {
    local model=$1 scale=$2
    local candidates=(
        "experiments/${model}_x${scale}/best.pt"
        "output/${model}_x${scale}/best.pt"
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
total=$(( ${#MODELS[@]} * ${#SCALES[@]} * ${#DATASETS[@]} ))
echo "============================================"
echo "  SISR 批量测试"
echo "============================================"
echo "  模型:   ${MODELS[*]}"
echo "  规模:   ${SCALES[*]}"
echo "  数据集: ${DATASETS[*]}"
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
            ((skipped++))
            continue
        fi

        for dataset in "${DATASETS[@]}"; do
            test_dir="data/datasets/${dataset}"
            if [[ ! -d "$test_dir" ]]; then
                echo "[SKIP] ${model}_x${scale} on ${dataset} — 数据集目录不存在: ${test_dir}"
                ((skipped++))
                continue
            fi

            out_dir="experiments/${model}_x${scale}/test/${dataset}"
            echo "--------------------------------------------"
            echo "[RUN]  ${model}_x${scale} on ${dataset}"
            echo "       ckpt: ${ckpt}"
            echo "       out:  ${out_dir}"

            cmd=(python test.py
                --ckpt "$ckpt"
                --test_dir "$test_dir"
                --model "$model"
                --scale "$scale"
                --out_dir "$out_dir"
            )
            if $SAVE_IMAGES; then
                cmd+=(--save_images)
            fi

            if "${cmd[@]}"; then
                echo "[DONE] ${model}_x${scale} on ${dataset}"
                ((passed++))
            else
                echo "[FAIL] ${model}_x${scale} on ${dataset}"
                ((failed++))
            fi
            echo ""
        done
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
