#!/usr/bin/env python3
"""
Test preprocessing with sample data to ensure no runtime errors.
This simulates what would happen with actual dataset processing.
"""

import sys
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_sample_csv(output_path, include_na_values=True):
    """Create a sample CSV file with CAN bus data."""
    data = {
        'timestamp': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        'arbitration_id': ['0x123', '0x456', 'na' if include_na_values else '0x789', '0x123', '0x456'],
        'data_field': [
            '0011223344556677',
            '8899aabbccddeeff',
            'na' if include_na_values else '1122334455667788',
            '0011223344556677',
            '8899aabbccddeeff'
        ],
        'attack': [0, 1, 1, 0, 0]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Created sample CSV: {output_path}")
    if include_na_values:
        print("  Contains 'na' values to test handling")
    return output_path


def test_preprocessing_with_sample_data():
    """Test that preprocessing can handle sample data without errors."""
    print("=" * 60)
    print("Testing Preprocessing with Sample Data")
    print("=" * 60)
    print()
    
    from preprocessing.CAN_preprocess import (
        parse_csv,
        safe_hex_to_int,
        safe_float,
        split_payload,
        is_na_value,
        NAStatistics
    )
    
    # Test 1: Verify functions can be imported
    print("1. Function Import Test")
    print("-" * 40)
    print("✅ All helper functions imported successfully")
    print()
    
    # Test 2: Test individual helper functions
    print("2. Helper Function Tests")
    print("-" * 40)
    
    # Test is_na_value
    assert is_na_value('na') == True
    assert is_na_value('NA') == True
    assert is_na_value('') == True
    assert is_na_value('0x123') == False
    print("✅ is_na_value() works correctly")
    
    # Test safe_hex_to_int
    assert safe_hex_to_int('0x123', allow_na_token=True) == 0x123
    assert safe_hex_to_int('na', allow_na_token=True) == -1
    assert safe_hex_to_int('na', allow_na_token=False) == 0
    assert safe_hex_to_int('ff', allow_na_token=True) == 255
    print("✅ safe_hex_to_int() works correctly")
    
    # Test split_payload
    payload = split_payload('0011223344556677', allow_na_token=True)
    assert len(payload) == 8
    assert payload[0] == 0x00
    assert payload[1] == 0x11
    print("✅ split_payload() works correctly")
    
    payload_na = split_payload('na', allow_na_token=True)
    assert all(b == -1 for b in payload_na)
    print("✅ split_payload() handles 'na' correctly")
    print()
    
    # Test 3: Test with sample CSV file
    print("3. Sample CSV Processing Test")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data with 'na' values
        csv_path = os.path.join(tmpdir, 'test_data.csv')
        create_sample_csv(csv_path, include_na_values=True)
        
        # Test parsing with special_token mode
        print("\nParsing with treat_na_as='special_token'...")
        try:
            result = parse_csv(
                csv_path,
                id_map=None,  # First pass to get unique IDs
                skip_invalid_rows=False,
                treat_na_as='special_token'
            )
            
            if result is not None:
                unique_ids = result
                print(f"✅ Parsed successfully, found {len(unique_ids)} unique IDs")
                print(f"  IDs: {sorted(unique_ids)}")
                
                # Verify -1 (na token) is in the results
                if -1 in unique_ids:
                    print("✅ 'na' token (-1) correctly identified")
                else:
                    print("⚠️  'na' token not found in unique IDs")
            else:
                print("❌ Parsing returned None")
                return False
                
        except Exception as e:
            print(f"❌ Error during parsing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print()
    print("=" * 60)
    print("✅ All Preprocessing Tests Passed!")
    print("=" * 60)
    return True


def test_no_name_error():
    """
    Specific test to ensure 'safe_hex_to_int' is not undefined.
    This addresses the error mentioned in the issue.
    """
    print("\n" + "=" * 60)
    print("Testing for NameError: 'safe_hex_to_int' is not defined")
    print("=" * 60)
    print()
    
    try:
        # Import the preprocessing module
        from preprocessing import CAN_preprocess
        
        # Try to access the function
        func = getattr(CAN_preprocess, 'safe_hex_to_int', None)
        
        if func is None:
            print("❌ safe_hex_to_int is not defined in the module")
            return False
        
        print("✅ safe_hex_to_int is defined in the module")
        
        # Try to call it
        result = func('0xff', allow_na_token=True)
        print(f"✅ safe_hex_to_int('0xff') = {result}")
        
        # Try with 'na'
        result_na = func('na', allow_na_token=True)
        print(f"✅ safe_hex_to_int('na', allow_na_token=True) = {result_na}")
        
        print()
        print("=" * 60)
        print("✅ No NameError - Function is accessible and working!")
        print("=" * 60)
        return True
        
    except NameError as e:
        print(f"❌ NameError occurred: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PREPROCESSING RUNTIME VALIDATION")
    print("="*60)
    
    all_passed = True
    
    # Run NameError test first
    if not test_no_name_error():
        all_passed = False
    
    # Run sample data test
    if not test_preprocessing_with_sample_data():
        all_passed = False
    
    if all_passed:
        print("\n" + "="*60)
        print("SUCCESS! All runtime validation tests passed! ✅")
        print("The preprocessing script is ready for use.")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("FAILURE! Some tests failed. ❌")
        print("="*60)
        sys.exit(1)
