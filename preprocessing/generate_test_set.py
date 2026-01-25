#!/usr/bin/env python3
"""
Generate Combined Test Data (Multi-Folder)

This script finds ALL test folders (test_01, test_02, etc.) in your dataset directory,
processes the CSVs using the training vocabulary, and merges them into a single
'test_data.npz' file so your evaluation script can read it.

Usage:
    python generate_test_set.py --dataset-dir /path/to/set_01 --output-dir /path/to/processed_output
"""

import os
import glob
import numpy as np
import argparse
import logging
from preprocess_core import parse_csv, create_windows, get_config

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(TEST)s - %(message)s')
logger = logging.getLogger(__name__)

def process_all_test_folders(dataset_dir, output_dir, window_size, stride, skip_invalid=False, treat_na='special_token'):
    
    # 1. LOAD EXISTING VOCABULARY
    id_map_path = os.path.join(output_dir, "id_map.npy")
    if not os.path.exists(id_map_path):
        logger.error(f"❌ Critical Error: id_map.npy not found in {output_dir}")
        logger.error("   You must run the training preprocessing first.")
        return

    logger.info(f"Loading ID Map from: {id_map_path}")
    id_map = np.load(id_map_path, allow_pickle=True).item()

    # 2. FIND ALL TEST FOLDERS
    # Looks for folders starting with "test_" inside set_01
    search_path = os.path.join(dataset_dir, "test_*")
    test_folders = sorted(glob.glob(search_path))
    
    # Filter only directories
    test_folders = [d for d in test_folders if os.path.isdir(d)]

    if not test_folders:
        logger.error(f"❌ No 'test_*' folders found in {dataset_dir}")
        return

    logger.info(f"Found {len(test_folders)} test scenarios:")
    for f in test_folders:
        logger.info(f"  - {os.path.basename(f)}")

    # 3. PROCESS ALL FILES
    X_test_ids = []
    X_test_pay = []
    X_test_time = []
    y_test = []
    
    total_files = 0

    for folder in test_folders:
        folder_name = os.path.basename(folder)
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        
        logger.info(f"\nProcessing {folder_name} ({len(csv_files)} files)...")
        
        for f in csv_files:
            # Parse CSV using the TRAIN id_map
            res = parse_csv(
                f, 
                id_map=id_map, 
                skip_invalid_rows=skip_invalid, 
                treat_na_as=treat_na
            )
            
            if res:
                ids, pay, time, lbl = res
                w_ids, w_pay, w_time, w_lbl = create_windows(ids, pay, time, lbl)

                if len(w_ids) > 0:
                    X_test_ids.append(w_ids)
                    X_test_pay.append(w_pay)
                    X_test_time.append(w_time)
                    y_test.append(w_lbl)
            
            total_files += 1

    # 4. SAVE MERGED OUTPUT
    if X_test_ids:
        output_file = os.path.join(output_dir, "test_data.npz")
        
        logger.info(f"\n{'='*40}")
        logger.info("Concatenating all test scenarios...")
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
        
        logger.info("✅ SUCCESS! test_data.npz generated.")
        logger.info(f"   Total Windows: {len(final_lbl)}")
        logger.info(f"   Attack Windows: {np.sum(final_lbl)}")
        logger.info(f"   Processed {total_files} files from {len(test_folders)} folders.")
    else:
        logger.warning("❌ No valid windows generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge multiple test folders into one test_data.npz')
    
    # Imports strictly for config defaults
    from preprocess_core import WINDOW_SIZE, STRIDE, get_config
    _, _, default_out, _, _ = get_config()

    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to the set_01 folder containing test_* subfolders')
    parser.add_argument('--output-dir', type=str, default=default_out,
                        help='Directory where id_map.npy is located')
    
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MULTI-FOLDER TEST GENERATOR")
    print(f"{'='*60}")
    
    process_all_test_folders(
        args.dataset_dir, 
        args.output_dir, 
        WINDOW_SIZE, 
        STRIDE
    )
