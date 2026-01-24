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
    
    # Test 'na' values - should return default
    assert safe_hex_to_int('na', default=0) == 0
    assert safe_hex_to_int('NA', default=0) == 0
    assert safe_hex_to_int('nan', default=0) == 0
    assert safe_hex_to_int('NaN', default=0) == 0
    assert safe_hex_to_int('', default=0) == 0
    assert safe_hex_to_int(pd.NA, default=0) == 0
    assert safe_hex_to_int(np.nan, default=0) == 0
    
    # Test custom default
    assert safe_hex_to_int('na', default=999) == 999
    
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
    result = split_payload('0011223344556677')
    assert result == [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77]
    
    # Test 'na' payload
    result = split_payload('na')
    assert result == [0, 0, 0, 0, 0, 0, 0, 0]
    
    # Test empty payload
    result = split_payload('')
    assert result == [0, 0, 0, 0, 0, 0, 0, 0]
    
    # Test short payload (should pad)
    result = split_payload('1122')
    assert result == [0x11, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    
    print("✅ split_payload tests passed")


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
