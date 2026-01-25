#!/usr/bin/env python3
"""
Generate Test Data (Step 2)

This script reuses the logic from your main preprocessing script to handle
the "Bucket #3" (Test Data). It ensures strict consistency by loading the
id_map generated during training.

Usage:
    python generate_test_set.py --test-dir /path/to/test_bucket --output-dir /path/to/processed_output
"""

import os
import glob
import numpy as np
import argparse
import logging
# IMPORTS FROM YOUR ORIGINAL SCRIPT
# Assumes you saved your script as 'CAN_preprocess.py'
from CAN_preprocess import parse_csv, create_windows, get_config, NAStatistics

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(TEST)s - %(message)s')
logger = logging.getLogger(__name__)

def process_test_data(test_dir, output_dir, window_size, stride, skip_invalid=False, treat_na='special_token'):
    """
    Process the test bucket using the EXISTING id_map.
    """
    
    # 1. LOAD EXISTING RESOURCES
    # We must use the exact same ID map from training
    id_map_path = os.path.join(output_dir, "id_map.npy")
    
    if not os.path.exists(id_map_path):
        logger.error(f"❌ Critical Error: id_map.npy not found in {output_dir}")
        logger.error("   You must run the training preprocessing (Step 1) before generating test data.")
        return

    logger.info(f"Loading ID Map from: {id_map_path}")
    id_map = np.load(id_map_path, allow_pickle=True).item()
    logger.info(f"Vocabulary size: {len(id_map)} IDs")

    # 2. LOCATE TEST FILES
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        return

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    logger.info(f"Found {len(test_files)} test CSVs in {test_dir}")

    if len(test_files) == 0:
        logger.error("No CSV files found! Check your path.")
        return

    # 3. PROCESS FILES
    X_test_ids = []
    X_test_pay = []
    X_test_time = []
    y_test = []

    logger.info("Processing test files...")
    
    for i, f in enumerate(test_files):
        # We pass the loaded id_map here. 
        # The parse_csv function will look up IDs in this map.
        # Any new IDs in the test set will be mapped to <UNK> automatically.
        res = parse_csv(
            f, 
            id_map=id_map, 
            skip_invalid_rows=skip_invalid, 
            treat_na_as=treat_na
        )
        
        if res:
            ids, pay, time, lbl = res
            
            # Apply Sliding Window
            w_ids, w_pay, w_time, w_lbl = create_windows(ids, pay, time, lbl) # Uses logic from core script

            if len(w_ids) > 0:
                X_test_ids.append(w_ids)
                X_test_pay.append(w_pay)
                X_test_time.append(w_time)
                y_test.append(w_lbl)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(test_files)} files...")

    # 4. SAVE OUTPUT
    if X_test_ids:
        output_file = os.path.join(output_dir, "test_data.npz")
        
        # Concatenate all windows
        logger.info("Concatenating data...")
        final_ids = np.concatenate(X_test_ids)
        final_pay = np.concatenate(X_test_pay)
        final_time = np.concatenate(X_test_time)
        final_lbl = np.concatenate(y_test)

        logger.info(f"Saving to {output_file}...")
        np.savez(output_file,
                 ids=final_ids,
                 payloads=final_pay,
                 deltas=final_time,
                 labels=final_lbl)
        
        logger.info("✅ Success! test_data.npz generated.")
        logger.info(f"   Total Samples: {len(final_lbl)}")
        logger.info(f"   Attack Samples: {np.sum(final_lbl)}")
    else:
        logger.warning("❌ No valid windows were generated from the test files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Test Data (.npz) from Raw CSVs')
    
    # Defaults taken from your existing config logic
    default_dataset, default_root, default_out, def_win, def_stride = get_config()
    
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to the folder containing raw Test CSVs')
    parser.add_argument('--output-dir', type=str, default=default_out,
                        help=f'Directory where id_map.npy is located (default: {default_out})')
    parser.add_argument('--skip-invalid', action='store_true', help='Skip invalid rows')
    
    args = parser.parse_args()

    # Reuse constants from the core script
    # We import these to ensure window size matches exactly what the model expects
    from CAN_preprocess import WINDOW_SIZE, STRIDE

    print(f"\n{'='*60}")
    print(f"GENERATING TEST SET (Step 2)")
    print(f"{'='*60}")
    print(f"Test Directory:   {args.test_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Window Size:      {WINDOW_SIZE}")
    print(f"Stride:           {STRIDE}")
    print(f"{'='*60}\n")

    process_test_data(
        args.test_dir, 
        args.output_dir, 
        WINDOW_SIZE, 
        STRIDE,
        skip_invalid=args.skip_invalid
    )
