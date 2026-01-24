#!/bin/bash
# Preprocess all datasets
# Runs preprocessing on all configured datasets

set -e

echo "============================================================"
echo "📊 Preprocessing All Datasets"
echo "============================================================"
echo ""

# Load dataset names from config
DATASETS=("set_01" "set_02" "set_03" "set_04")

COUNTER=1
TOTAL=${#DATASETS[@]}

for dataset in "${DATASETS[@]}"; do
    echo "[$COUNTER/$TOTAL] Processing $dataset..."
    echo "------------------------------------------------------------"
    
    # Check if raw data exists
    RAW_DATA_PATH="/workspace/data/can-train-and-test-v1.5/$dataset"
    
    if [ ! -d "$RAW_DATA_PATH" ]; then
        echo "⚠️  Raw data not found: $RAW_DATA_PATH"
        echo "   Skipping $dataset"
        echo ""
        COUNTER=$((COUNTER + 1))
        continue
    fi
    
    # Run preprocessing
    DATASET=$dataset python preprocessing/CAN_preprocess.py
    
    echo ""
    echo "✅ Completed $dataset"
    echo ""
    
    COUNTER=$((COUNTER + 1))
done

echo "============================================================"
echo "✅ All datasets preprocessed!"
echo "============================================================"
echo ""
echo "📁 Processed data saved to:"
echo "   /workspace/data/processed_data/"
echo ""
echo "📋 Next steps:"
echo "   1. Train on single dataset:"
echo "      python train.py"
echo ""
echo "   2. Run cross-dataset evaluation:"
echo "      python scripts/cross_dataset_eval.py --all"
echo ""
echo "   3. Train on combined datasets:"
echo "      python scripts/train_combined.py --datasets set_01,set_02,set_03,set_04"
echo ""
