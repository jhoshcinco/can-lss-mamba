"""Tests for preprocessing functions."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


def test_safe_hex_to_int():
    """Test safe_hex_to_int function handles various inputs."""
    # Import the function
    from preprocessing.CAN_preprocess import safe_hex_to_int
    
    # Test valid hex values
    assert safe_hex_to_int('ff', default=0) == 255
    assert safe_hex_to_int('0xff', default=0) == 255
    assert safe_hex_to_int('FF', default=0) == 255
    assert safe_hex_to_int('0x199', default=0) == 409
    assert safe_hex_to_int('199', default=0) == 409
    
    # Test 'na' values with allow_na_token=False (old behavior) - should return default
    assert safe_hex_to_int('na', default=0, allow_na_token=False) == 0
    assert safe_hex_to_int('NA', default=0, allow_na_token=False) == 0
    assert safe_hex_to_int('nan', default=0, allow_na_token=False) == 0
    assert safe_hex_to_int('NaN', default=0, allow_na_token=False) == 0
    assert safe_hex_to_int('', default=0, allow_na_token=False) == 0
    assert safe_hex_to_int(pd.NA, default=0, allow_na_token=False) == 0
    assert safe_hex_to_int(np.nan, default=0, allow_na_token=False) == 0
    
    # Test 'na' values with allow_na_token=True (new behavior) - should return -1
    assert safe_hex_to_int('na', default=0, allow_na_token=True) == -1
    assert safe_hex_to_int('NA', default=0, allow_na_token=True) == -1
    assert safe_hex_to_int('nan', default=0, allow_na_token=True) == -1
    assert safe_hex_to_int('NaN', default=0, allow_na_token=True) == -1
    assert safe_hex_to_int('', default=0, allow_na_token=True) == -1
    assert safe_hex_to_int(pd.NA, default=0, allow_na_token=True) == -1
    assert safe_hex_to_int(np.nan, default=0, allow_na_token=True) == -1
    
    # Test custom default with allow_na_token=False
    assert safe_hex_to_int('na', default=999, allow_na_token=False) == 999
    
    # Test invalid values
    assert safe_hex_to_int('invalid', default=0) == 0
    assert safe_hex_to_int('xyz', default=0) == 0
    
    print("✅ safe_hex_to_int tests passed")


def test_safe_float():
    """Test safe_float function handles various inputs."""
    from preprocessing.CAN_preprocess import safe_float
    
    # Test valid float values
    assert safe_float('1.5', default=0.0) == 1.5
    assert safe_float('123.456', default=0.0) == 123.456
    assert safe_float(1.5, default=0.0) == 1.5
    
    # Test integer values
    assert safe_float('123', default=0.0) == 123.0
    assert safe_float(123, default=0.0) == 123.0
    
    # Test 'na' values - should return default
    assert safe_float('na', default=0.0) == 0.0
    assert safe_float('NA', default=0.0) == 0.0
    assert safe_float('nan', default=0.0) == 0.0
    assert safe_float('NaN', default=0.0) == 0.0
    assert safe_float('', default=0.0) == 0.0
    assert safe_float(pd.NA, default=0.0) == 0.0
    assert safe_float(np.nan, default=0.0) == 0.0
    
    # Test custom default
    assert safe_float('na', default=999.0) == 999.0
    
    # Test invalid values
    assert safe_float('invalid', default=0.0) == 0.0
    assert safe_float('xyz', default=0.0) == 0.0
    
    print("✅ safe_float tests passed")


def test_validate_dataframe():
    """Test validate_dataframe function."""
    from preprocessing.CAN_preprocess import validate_dataframe
    
    # Test valid dataframe
    df_valid = pd.DataFrame({
        'timestamp': [1.0, 2.0, 3.0],
        'arbitration_id': ['0x123', '0x456', '0x789'],
        'data_field': ['0011223344556677', '8899aabbccddeeff', '1122334455667788'],
        'attack': [0, 0, 1]
    })
    
    assert validate_dataframe(df_valid, "test.csv") == True
    
    # Test dataframe with 'na' values - should still return True but log warnings
    df_with_na = pd.DataFrame({
        'timestamp': [1.0, 'na', 3.0],
        'arbitration_id': ['0x123', 'na', '0x789'],
        'data_field': ['0011223344556677', 'na', '1122334455667788'],
        'attack': [0, 0, 1]
    })
    
    # Should still return True (we handle na values now)
    assert validate_dataframe(df_with_na, "test_na.csv") == True
    
    # Test dataframe with missing columns
    df_missing = pd.DataFrame({
        'timestamp': [1.0, 2.0, 3.0],
        'arbitration_id': ['0x123', '0x456', '0x789']
    })
    
    # Should return False due to missing columns
    assert validate_dataframe(df_missing, "test_missing.csv") == False
    
    print("✅ validate_dataframe tests passed")


def test_split_payload_with_na():
    """Test that split_payload handles 'na' values correctly."""
    from preprocessing.CAN_preprocess import split_payload
    
    # Test valid payload
    result = split_payload('0011223344556677', allow_na_token=False)
    assert result == [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77]
    
    # Test 'na' payload with allow_na_token=False (old behavior)
    result = split_payload('na', allow_na_token=False)
    assert result == [0, 0, 0, 0, 0, 0, 0, 0]
    
    # Test 'na' payload with allow_na_token=True (new behavior)
    result = split_payload('na', allow_na_token=True)
    assert result == [-1, -1, -1, -1, -1, -1, -1, -1]
    
    # Test empty payload with allow_na_token=False
    result = split_payload('', allow_na_token=False)
    assert result == [0, 0, 0, 0, 0, 0, 0, 0]
    
    # Test empty payload with allow_na_token=True
    result = split_payload('', allow_na_token=True)
    assert result == [-1, -1, -1, -1, -1, -1, -1, -1]
    
    # Test short payload (should pad)
    result = split_payload('1122', allow_na_token=False)
    assert result == [0x11, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    
    print("✅ split_payload tests passed")


def test_na_statistics():
    """Test NAStatistics class tracking."""
    from preprocessing.CAN_preprocess import NAStatistics
    import logging
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Create statistics tracker
    stats = NAStatistics()
    
    # Test initial values
    assert stats.total_rows == 0
    assert stats.na_in_can_id == 0
    assert stats.na_in_dlc == 0
    assert stats.na_in_data_bytes == 0
    assert stats.rows_with_any_na == 0
    assert stats.skipped_rows == 0
    
    # Simulate tracking
    stats.total_rows = 100
    stats.na_in_can_id = 10
    stats.na_in_data_bytes = 5
    stats.rows_with_any_na = 12
    
    # Test logging doesn't crash
    stats.log_statistics("test.csv", logger)
    
    print("✅ NAStatistics tests passed")


def test_vocabulary_with_na_token():
    """Test that vocabulary mapping correctly includes -1 token."""
    # Simulate vocabulary building with -1 (na token)
    unique_ids = [-1, 0, 100, 200, 300]
    
    # Build map with special token handling
    id_map = {}
    id_map[-1] = 0  # Reserve index 0 for 'na' token
    
    idx = 1
    for can_id in sorted(unique_ids):
        if can_id != -1:
            id_map[can_id] = idx
            idx += 1
    
    id_map['<UNK>'] = len(id_map)
    
    # Verify mapping
    assert id_map[-1] == 0, "'na' token should be at index 0"
    assert id_map[0] == 1, "CAN ID 0 should be at index 1"
    assert id_map[100] == 2, "CAN ID 100 should be at index 2"
    assert '<UNK>' in id_map, "Unknown token should be in map"
    
    print("✅ Vocabulary with na token tests passed")


def test_treat_na_as_modes():
    """Test different treat_na_as modes."""
    from preprocessing.CAN_preprocess import safe_hex_to_int, split_payload
    
    # Mode: special_token (allow_na_token=True)
    assert safe_hex_to_int('na', default=0, allow_na_token=True) == -1
    assert split_payload('na', allow_na_token=True) == [-1] * 8
    
    # Mode: zero (allow_na_token=False) 
    assert safe_hex_to_int('na', default=0, allow_na_token=False) == 0
    assert split_payload('na', allow_na_token=False) == [0] * 8
    
    # Mode: skip would be handled at dataframe level
    
    print("✅ treat_na_as modes tests passed")


def run_all_tests():
    """Run all preprocessing tests."""
    print("="*60)
    print("Running Preprocessing Tests")
    print("="*60)
    print()
    
    tests = [
        ("safe_hex_to_int", test_safe_hex_to_int),
        ("safe_float", test_safe_float),
        ("validate_dataframe", test_validate_dataframe),
        ("split_payload_with_na", test_split_payload_with_na),
        ("na_statistics", test_na_statistics),
        ("vocabulary_with_na_token", test_vocabulary_with_na_token),
        ("treat_na_as_modes", test_treat_na_as_modes),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 40)
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            failed += 1
    
    print()
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
