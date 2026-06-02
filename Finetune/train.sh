#!/usr/bin/env bash
# Run the LoRA fine-tune on the EXACT 2026 training set.
#
# Defaults are sized for the RTX 5070 (12 GB GDDR7, Blackwell sm_120).
# Bigger card (16 GB)? BATCH=4 GRAD_ACCUM=4. (24 GB+)? BATCH=8 GRAD_ACCUM=2
# GRAD_CKPT=0 for faster wall time. Still OOMing on 12 GB? BATCH=1 GRAD_ACCUM=16.

set -euo pipefail

# shellcheck disable=SC1091
source .venv/bin/activate

# Must match the model setup.sh downloaded (default Qwen/Qwen3.5-4B-Base).
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-4B-Base}
DATA=${DATA:-data/annotation_ready_merged.json}
OUT=${OUT:-artifacts/translator-lora}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
LORA_R=${LORA_R:-32}
LR=${LR:-2e-4}
# Memory-safety: keep both ON on 12 GB (FA2 + grad-checkpointing). On 24 GB+ you
# can set GRAD_CKPT=0 for ~25% faster wall time.
GRAD_CKPT=${GRAD_CKPT:-1}
ATTN=${ATTN:-flash_attention_2}
# Set REQUIRE_GOAL=1 after manual goal-FOL annotation is complete; the trainer
# will drop any record that doesn't have an annotated questions-FOL entry.
REQUIRE_GOAL=${REQUIRE_GOAL:-0}

mkdir -p artifacts

if [ "$GRAD_CKPT" = "1" ]; then
    GC_FLAG="--gradient-checkpointing"
else
    GC_FLAG="--no-gradient-checkpointing"
fi

if [ "$REQUIRE_GOAL" = "1" ]; then
    GOAL_FLAG="--require-goal-fol"
else
    GOAL_FLAG="--allow-placeholder-goal"
fi

echo "== Fine-tuning $MODEL_ID (LoRA r=$LORA_R) =="
echo "  data:       $DATA"
echo "  output:     $OUT"
echo "  epochs:     $EPOCHS"
echo "  batch:      $BATCH  (grad_accum=$GRAD_ACCUM → effective $((BATCH * GRAD_ACCUM)))"
echo "  max_seq:    $MAX_SEQ_LEN"
echo "  grad_ckpt:  $GRAD_CKPT     attn: $ATTN"
echo "  lr:         $LR"
echo

python -m finetune.train_lora \
    --train "$DATA" \
    --out "$OUT" \
    --base-model "$MODEL_ID" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lora-r "$LORA_R" \
    --lr "$LR" \
    $GC_FLAG \
    $GOAL_FLAG \
    --attn "$ATTN"

echo
echo "Done. LoRA adapter is in $OUT/. To ship back:"
echo "  tar -czf translator-lora.tar.gz -C artifacts translator-lora"
