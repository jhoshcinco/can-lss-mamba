#!/bin/bash
# Quick hyperparameter test (3-5 experiments)
# Tests a few key hyperparameters to get quick feedback

set -e

echo "============================================================"
echo "🔬 Quick Hyperparameter Test"
echo "============================================================"
echo ""
echo "Testing 3 learning rates with fixed batch_size=32, epochs=20"
echo "This gives quick feedback on hyperparameter sensitivity."
echo ""
echo "⚠️  Uses VALIDATION metrics (Bucket 2) for comparison."
echo "   Test metrics (Bucket 3) are NOT used - no data leakage!"
echo ""

# Configuration
DATASET=${DATASET:-set_01}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-20}
WANDB_ENABLED=${WANDB_ENABLED:-true}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  WandB: $WANDB_ENABLED"
echo ""
echo "============================================================"
echo ""

# Test learning rates
LR_VALUES=(0.0001 0.0005 0.001)
COUNTER=1
TOTAL=${#LR_VALUES[@]}

for lr in "${LR_VALUES[@]}"; do
    echo "[$COUNTER/$TOTAL] Testing LR=$lr"
    echo "------------------------------------------------------------"
    
    # Use unique checkpoint directory per learning rate
    OUT_DIR=/workspace/checkpoints/quick_test_lr${lr} \
    DATASET=$DATASET \
    BATCH_SIZE=$BATCH_SIZE \
    EPOCHS=$EPOCHS \
    LR=$lr \
    WANDB_ENABLED=$WANDB_ENABLED \
    WANDB_PROJECT=can-lss-mamba \
    WANDB_ENTITY=jhoshcinco-ca-western-university \
    WANDB_TAGS="quick_test,lr_sweep" \
    WANDB_NAME="quick_test_lr_${lr}" \
    python train.py
    
    echo ""
    echo "✅ Completed LR=$lr"
    echo ""
    
    COUNTER=$((COUNTER + 1))
done

echo "============================================================"
echo "✅ Quick test complete!"
echo "============================================================"
echo ""
echo "📊 View results at:"
echo "   https://wandb.ai/jhoshcinco-ca-western-university/can-lss-mamba"
echo ""
echo "📋 Compare runs with:"
echo "   python scripts/compare_runs.py --tag quick_test"
echo ""
echo "💡 Tip: The best learning rate from this test can be used for"
echo "   more extensive hyperparameter tuning with grid_search.py"
echo ""
