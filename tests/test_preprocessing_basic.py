"""Basic test to verify preprocessing script syntax and structure."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_script_syntax():
    """Test that the preprocessing script has valid syntax."""
    import py_compile
    
    preprocessing_script = project_root / "preprocessing" / "CAN_preprocess.py"
    
    try:
        py_compile.compile(str(preprocessing_script), doraise=True)
        print(f"✅ Syntax check passed for {preprocessing_script}")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error in {preprocessing_script}: {e}")
        return False


def test_functions_exist():
    """Test that required functions exist in the script."""
    preprocessing_script = project_root / "preprocessing" / "CAN_preprocess.py"
    
    with open(preprocessing_script, 'r') as f:
        content = f.read()
    
    required_functions = [
        'safe_hex_to_int',
        'safe_float',
        'validate_dataframe',
        'parse_csv',
        'create_windows',
        'run_pipeline',
        'parse_args'
    ]
    
    missing = []
    for func in required_functions:
        if f"def {func}" not in content:
            missing.append(func)
    
    if missing:
        print(f"❌ Missing functions: {', '.join(missing)}")
        return False
    else:
        print(f"✅ All required functions found")
        return True


def test_safe_conversion_logic():
    """Test safe conversion logic without importing dependencies."""
    # Test that the logic handles 'na' values correctly
    
    def is_na_value_test(value):
        """Test implementation of is_na_value."""
        # Using a simplified check that doesn't require pandas
        if value is None:
            return True
        if isinstance(value, str) and value.lower() in ['na', 'nan', '']:
            return True
        # Note: Can't test pd.isna() without pandas import
        return False
    
    def safe_hex_to_int_test(value, default=0):
        """Test implementation of safe_hex_to_int."""
        if is_na_value_test(value):
            return default
        
        try:
            if isinstance(value, str):
                value = value.strip().lower()
                if value.startswith('0x'):
                    return int(value, 16)
                else:
                    return int(value, 16)
            return int(value)
        except (ValueError, TypeError):
            return default
    
    # Test cases
    test_cases = [
        ('ff', 0, 255),
        ('0xff', 0, 255),
        ('na', 0, 0),
        ('NA', 0, 0),
        ('', 0, 0),
        ('na', 999, 999),
        ('199', 0, 409),
        (None, 0, 0),
    ]
    
    all_passed = True
    for value, default, expected in test_cases:
        result = safe_hex_to_int_test(value, default)
        if result != expected:
            print(f"❌ safe_hex_to_int_test('{value}', {default}) = {result}, expected {expected}")
            all_passed = False
    
    if all_passed:
        print("✅ Safe conversion logic tests passed")
    
    return all_passed


def test_command_line_args():
    """Test that command-line arguments are defined."""
    preprocessing_script = project_root / "preprocessing" / "CAN_preprocess.py"
    
    with open(preprocessing_script, 'r') as f:
        content = f.read()
    
    required_args = [
        '--skip-invalid-rows',
        '--dataset',
        '--dataset-root',
        '--output-dir'
    ]
    
    missing = []
    for arg in required_args:
        if arg not in content:
            missing.append(arg)
    
    if missing:
        print(f"❌ Missing command-line arguments: {', '.join(missing)}")
        return False
    else:
        print(f"✅ All required command-line arguments found")
        return True


def test_logging_infrastructure():
    """Test that logging is properly set up."""
    preprocessing_script = project_root / "preprocessing" / "CAN_preprocess.py"
    
    with open(preprocessing_script, 'r') as f:
        content = f.read()
    
    # Check for logging setup
    checks = [
        ('import logging', 'Logging module imported'),
        ('logging.basicConfig', 'Logging configured'),
        ('logger = logging.getLogger', 'Logger created'),
        ('logger.info', 'Info logging used'),
        ('logger.error', 'Error logging used'),
        ('logger.warning', 'Warning logging used'),
    ]
    
    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}")
            all_passed = False
    
    if all_passed:
        print("✅ Logging infrastructure tests passed")
    
    return all_passed


def test_data_quality_features():
    """Test that data quality tracking features exist."""
    preprocessing_script = project_root / "preprocessing" / "CAN_preprocess.py"
    
    with open(preprocessing_script, 'r') as f:
        content = f.read()
    
    # Check for data quality features
    features = [
        ('stats = {', 'Statistics tracking'),
        ("'invalid_can_id'", 'Invalid CAN ID tracking'),
        ("'invalid_timestamp'", 'Invalid timestamp tracking'),
        ("'invalid_data_bytes'", 'Invalid data bytes tracking'),
        ('data_quality_report.json', 'Data quality report generation'),
        ('validate_dataframe', 'DataFrame validation'),
    ]
    
    all_passed = True
    for feature_str, description in features:
        if feature_str in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}")
            all_passed = False
    
    if all_passed:
        print("✅ Data quality features tests passed")
    
    return all_passed


def run_all_tests():
    """Run all basic tests."""
    print("="*60)
    print("Running Basic Preprocessing Tests")
    print("="*60)
    print()
    
    tests = [
        ("Script Syntax", test_script_syntax),
        ("Required Functions", test_functions_exist),
        ("Safe Conversion Logic", test_safe_conversion_logic),
        ("Command-line Arguments", test_command_line_args),
        ("Logging Infrastructure", test_logging_infrastructure),
        ("Data Quality Features", test_data_quality_features),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
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
    success = run_all_tests()
    sys.exit(0 if success else 1)
