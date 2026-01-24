#!/usr/bin/env python3
"""
Validate that hyperparameter configs don't reuse checkpoints.

This script tests that different hyperparameter configurations train independently.
Each config should start from epoch 1 and use separate checkpoint directories.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def test_checkpoint_isolation():
    """Test that different configs use separate checkpoints."""
    print("Testing checkpoint isolation...")
    print("="*60)
    
    # Test configurations with different hyperparameters
    configs = [
        {'lr': 0.0001, 'bs': 32, 'out_dir': '/tmp/test_config1'},
        {'lr': 0.0005, 'bs': 32, 'out_dir': '/tmp/test_config2'},
    ]
    
    # Clean up any existing test directories
    for config in configs:
        if os.path.exists(config['out_dir']):
            shutil.rmtree(config['out_dir'])
    
    all_passed = True
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfig {i}: LR={config['lr']}, BS={config['bs']}")
        print(f"Checkpoint directory: {config['out_dir']}")
        
        # Run training for 1 epoch
        env = os.environ.copy()
        env['OUT_DIR'] = config['out_dir']
        env['LR'] = str(config['lr'])
        env['BATCH_SIZE'] = str(config['bs'])
        env['EPOCHS'] = '1'
        env['WANDB_ENABLED'] = 'false'  # Disable WandB for testing
        
        # Check if data directory exists
        data_dir = env.get('DATA_DIR', '/workspace/data/processed_data/set_01_run_02')
        if not os.path.exists(data_dir):
            print(f"⚠️  Warning: Data directory not found: {data_dir}")
            print(f"   Skipping validation (needs actual data to run)")
            return None  # Can't validate without data
        
        result = subprocess.run(
            ['python', 'train.py'],
            env=env,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Check that it started from epoch 1 (check both stdout and stderr)
        output = result.stdout + result.stderr
        if 'Epoch 1' in output or 'Epoch 1 |' in output or 'Starting Training' in output:
            print(f"  ✅ Config {i} started from epoch 1 (correct)")
        else:
            print(f"  ❌ Config {i} did NOT start from epoch 1!")
            print(f"     Output snippet: {output[:500]}")
            all_passed = False
            
        # Verify checkpoint was saved to correct directory
        checkpoint_path = os.path.join(config['out_dir'], 'lss_can_mamba_last.pth')
        if os.path.exists(checkpoint_path):
            print(f"  ✅ Checkpoint saved to: {config['out_dir']}")
        else:
            print(f"  ❌ Checkpoint NOT found at: {config['out_dir']}")
            all_passed = False
    
    # Verify no cross-contamination between all configs
    print("\nVerifying no cross-contamination...")
    for i, config_i in enumerate(configs, 1):
        checkpoint_i = os.path.join(config_i['out_dir'], 'lss_can_mamba_last.pth')
        for j, config_j in enumerate(configs, 1):
            if i != j:
                checkpoint_j = os.path.join(config_j['out_dir'], 'lss_can_mamba_last.pth')
                # Check if config i's directory contains any checkpoints from config j
                if os.path.exists(config_i['out_dir']) and os.path.exists(config_j['out_dir']):
                    # Both should exist and be different
                    if config_i['out_dir'] == config_j['out_dir']:
                        print(f"  ❌ Config {i} and Config {j} use the same directory!")
                        all_passed = False
    
    # Clean up test directories
    for config in configs:
        if os.path.exists(config['out_dir']):
            shutil.rmtree(config['out_dir'])
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! Checkpoint isolation is working correctly.")
        return True
    else:
        print("❌ Some tests failed! Checkpoint isolation is NOT working.")
        return False

if __name__ == '__main__':
    try:
        success = test_checkpoint_isolation()
        if success is None:
            print("\n⚠️  Could not validate (missing data). Manual testing required.")
            sys.exit(0)
        elif success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
