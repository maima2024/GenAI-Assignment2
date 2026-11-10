#!/usr/bin/env bash
# End-to-end pipeline (≈60-80 min on one free Kaggle/Colab T4).  Re-running skips finished stages.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # single GPU: avoids DataParallel on 2xT4
export TOKENIZERS_PARALLELISM=false
TARGET="${TARGET:-Qwen/Qwen2.5-1.5B-Instruct}"
DRAFT="${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}"
EPOCHS="${EPOCHS:-2}"
mkdir -p results/logs results/eval results/bench runs

stage() { echo; echo "=================== $1 ==================="; date; }

stage "environment"
python -m trialmatch.modeling | tee results/env.json

stage "split + manifest + leakage checks"
[ -f results/split/split_manifest.json ] || python -m trialmatch.split --out results/split 2>&1 | tee results/logs/split.log

stage "hyper-parameter sweep on the 0.5B model (validation only)"
for cfg in "16 2e-4" "16 5e-5" "8 2e-4"; do
  set -- $cfg; R=$1; LR=$2; NAME="r${R}_lr${LR}"
  if [ ! -f "runs/sweep_${NAME}/adapter/adapter_config.json" ]; then
    python -m trialmatch.train --model "$DRAFT" --out "runs/sweep_${NAME}" --lora_r "$R" --lora_alpha $((2*R)) \
      --lr "$LR" --epochs "$EPOCHS" --batch 4 --grad_accum 4 2>&1 | tee "results/logs/train_sweep_${NAME}.log"
  fi
  [ -f "results/eval/sweep_${NAME}_val/metrics.json" ] || python -m trialmatch.evaluate --model "$DRAFT" \
      --adapter "runs/sweep_${NAME}/adapter" --split val --out "results/eval/sweep_${NAME}_val" 2>&1 | tee "results/logs/eval_sweep_${NAME}_val.log"
done
[ -f results/eval/draft_base_val/metrics.json ] || python -m trialmatch.evaluate --model "$DRAFT" --split val \
    --out results/eval/draft_base_val 2>&1 | tee results/logs/eval_draft_base_val.log
BEST=$(python scripts/select_config.py results/eval/sweep_*_val --out results/selection.json | tail -1)
echo "selected config: $BEST"
R=$(echo "$BEST" | sed -E 's/^r([0-9]+)_lr.*/\1/'); LR=$(echo "$BEST" | sed -E 's/^r[0-9]+_lr(.*)$/\1/')
DRAFT_ADAPTER="runs/sweep_${BEST}/adapter"

stage "fine-tune target ($TARGET) with r=$R lr=$LR"
[ -f runs/target/adapter/adapter_config.json ] || python -m trialmatch.train --model "$TARGET" --out runs/target \
    --lora_r "$R" --lora_alpha $((2*R)) --lr "$LR" --epochs "$EPOCHS" --batch 2 --grad_accum 8 2>&1 | tee results/logs/train_target.log
cp runs/target/train_log.json results/train_log_target.json
for d in runs/sweep_*; do cp "$d/train_log.json" "results/train_log_$(basename $d).json"; done

stage "before / after fine-tuning: validation, then the untouched test split"
for SPLIT in val test; do
  [ -f results/eval/base_${SPLIT}/metrics.json ] || python -m trialmatch.evaluate --model "$TARGET" --split $SPLIT \
      --out results/eval/base_${SPLIT} 2>&1 | tee results/logs/eval_base_${SPLIT}.log
  [ -f results/eval/ft_${SPLIT}/metrics.json ] || python -m trialmatch.evaluate --model "$TARGET" --adapter runs/target/adapter \
      --split $SPLIT --out results/eval/ft_${SPLIT} 2>&1 | tee results/logs/eval_ft_${SPLIT}.log
done

stage "Task 4: attention"
[ -f results/bench/attention.json ] || python -m benchmarks.bench_attention --out results/bench/attention.json 2>&1 | tee results/logs/bench_attention.log

stage "Task 3: KV cache"
[ -f results/bench/kv.json ] || python -m benchmarks.bench_kv --model "$TARGET" --adapter runs/target/adapter \
    --fp32_check --out results/bench/kv.json 2>&1 | tee results/logs/bench_kv.log

stage "Task 5: speculative decoding"
[ -f results/bench/speculative.json ] || python -m benchmarks.bench_speculative --target "$TARGET" \
    --target_adapter runs/target/adapter --draft "$DRAFT" --draft_adapter "$DRAFT_ADAPTER" \
    --out results/bench/speculative.json 2>&1 | tee results/logs/bench_speculative.log

stage "figures, error analysis, bundle"
python scripts/make_figures.py 2>&1 | tee results/logs/figures.log || echo "figure step failed (non-fatal)"
python scripts/error_analysis.py 2>&1 | tee results/logs/error_analysis.log || echo "error analysis failed (non-fatal)"
mkdir -p adapters_export
cp -r runs/target/adapter adapters_export/target_qwen2.5-1.5b_lora
cp -r "$DRAFT_ADAPTER" adapters_export/draft_qwen2.5-0.5b_lora
echo "{\"target_adapter\": \"adapters_export/target_qwen2.5-1.5b_lora\", \"draft_adapter\": \"adapters_export/draft_qwen2.5-0.5b_lora\", \"selected\": \"$BEST\"}" > results/adapters.json
date
echo "ALL DONE"
