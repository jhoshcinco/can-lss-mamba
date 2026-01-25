#!/usr/bin/env python3
"""Verify that all int(x, 16) calls have been replaced."""

import re
import sys
from pathlib import Path

# Get the preprocessing script path
project_root = Path(__file__).parent.parent
preprocessing_script = project_root / 'preprocessing' / 'CAN_preprocess.py'

with open(preprocessing_script, 'r') as f:
    content = f.read()

print("=" * 60)
print("Verifying Preprocessing Fix")
print("=" * 60)
print()

all_checks_passed = True

# Find any remaining int(x, 16) patterns
print("1. Checking for int(x, 16) calls...")
print("-" * 40)
remaining = re.findall(r'int\([^,]+,\s*16\)', content)

if remaining:
    # Filter out the ones inside safe_hex_to_int function (which are correct)
    lines = content.split('\n')
    problematic_calls = []
    
    in_safe_hex_function = False
    for i, line in enumerate(lines):
        if 'def safe_hex_to_int' in line:
            in_safe_hex_function = True
        elif in_safe_hex_function and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            # We've left the function
            in_safe_hex_function = False
        
        if 'int(' in line and ', 16)' in line:
            if not in_safe_hex_function:
                problematic_calls.append((i + 1, line.strip()))
    
    if problematic_calls:
        print("❌ Found int(x, 16) calls outside safe_hex_to_int that need replacement:")
        for line_num, line in problematic_calls:
            print(f"  Line {line_num}: {line}")
        all_checks_passed = False
    else:
        print("✅ All int(x, 16) calls are within safe_hex_to_int (correct)")
else:
    print("✅ No int(x, 16) calls found (all replaced)")

# Verify safe_hex_to_int exists and is at module level
print()
print("2. Checking safe_hex_to_int function...")
print("-" * 40)

if 'def safe_hex_to_int' not in content:
    print("❌ safe_hex_to_int function not found")
    all_checks_passed = False
else:
    # Check that function is not indented
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def safe_hex_to_int' in line:
            if line.startswith(' ') or line.startswith('\t'):
                print(f"❌ safe_hex_to_int is indented at line {i+1} (should be at column 0)")
                all_checks_passed = False
            else:
                print(f"✅ safe_hex_to_int is at module level (line {i+1})")
            break

# Check that function is defined before first use
print()
print("3. Checking function definition order...")
print("-" * 40)

func_def_line = -1
first_use_line = -1

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'def safe_hex_to_int' in line and func_def_line == -1:
        func_def_line = i
    elif 'safe_hex_to_int(' in line and 'def safe_hex_to_int' not in line and first_use_line == -1:
        first_use_line = i

if first_use_line != -1 and func_def_line != -1 and first_use_line < func_def_line:
    print(f"❌ safe_hex_to_int used at line {first_use_line+1} but defined at line {func_def_line+1}")
    all_checks_passed = False
else:
    if func_def_line != -1:
        print(f"✅ safe_hex_to_int defined at line {func_def_line+1} before first use")
    else:
        print("⚠️  Could not determine usage order")

# Check that other helper functions are at module level
print()
print("4. Checking other helper functions...")
print("-" * 40)

helper_functions = ['is_na_value', 'safe_float', 'split_payload']
for func_name in helper_functions:
    if f'def {func_name}' not in content:
        print(f"❌ {func_name} function not found")
        all_checks_passed = False
    else:
        for i, line in enumerate(lines):
            if f'def {func_name}' in line:
                if line.startswith(' ') or line.startswith('\t'):
                    print(f"❌ {func_name} is indented at line {i+1} (should be at column 0)")
                    all_checks_passed = False
                else:
                    print(f"✅ {func_name} is at module level (line {i+1})")
                break

# Verify split_payload uses safe_hex_to_int
print()
print("5. Checking split_payload uses safe_hex_to_int...")
print("-" * 40)

in_split_payload = False
uses_safe_hex = False
split_payload_start = -1

for i, line in enumerate(lines):
    if 'def split_payload' in line:
        in_split_payload = True
        split_payload_start = i
    elif in_split_payload:
        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            # Left the function
            break
        if 'safe_hex_to_int' in line:
            uses_safe_hex = True

if uses_safe_hex:
    print(f"✅ split_payload uses safe_hex_to_int")
else:
    print(f"❌ split_payload does not use safe_hex_to_int")
    all_checks_passed = False

# Summary
print()
print("=" * 60)
if all_checks_passed:
    print("✅ All checks passed! Preprocessing fix is correct.")
    print("=" * 60)
    sys.exit(0)
else:
    print("❌ Some checks failed. Please review the issues above.")
    print("=" * 60)
    sys.exit(1)
