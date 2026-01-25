# NA Value Handling Implementation Summary

## Overview

This implementation adds proper handling of 'na' values in CAN dataset preprocessing, treating them as special tokens that preserve attack signature information instead of destroying it by replacing with 0.

## Key Changes

### 1. Command-Line Interface

Added `--treat-na-as` argument with three modes:

```bash
# Default: Treat 'na' as special attack indicator (RECOMMENDED)
python preprocessing/CAN_preprocess.py --treat-na-as special_token

# Alternative: Replace with 0 (old behavior, not recommended)
python preprocessing/CAN_preprocess.py --treat-na-as zero

# Alternative: Skip rows with 'na' (if measurement error)
python preprocessing/CAN_preprocess.py --treat-na-as skip
```

### 2. Core Functions Updated

#### safe_hex_to_int()
```python
def safe_hex_to_int(value, default=0, allow_na_token=False):
    """
    Args:
        allow_na_token: If True, 'na' returns -1 (special token)
                       If False, 'na' returns default
    """
```

#### split_payload()
```python
def split_payload(hex_str, allow_na_token=False):
    """
    Args:
        allow_na_token: If True, 'na' bytes become -1, else 0
    """
```

### 3. Vocabulary Building

When `treat_na_as='special_token'`:
- Vocab index 0 is reserved for -1 (the 'na' token)
- Normal CAN IDs start from index 1
- Model can learn that index 0 indicates attacks

**Example:**
```
Attack 'na':       -1 → vocab[0]  (attack indicator)
CAN ID 0x000:       0 → vocab[1]  (normal traffic)
CAN ID 0x100:     256 → vocab[2]  (normal traffic)
```

### 4. Statistics Tracking

New `NAStatistics` class tracks:
- Total rows processed
- 'na' in CAN IDs (with percentage)
- 'na' in DLC
- 'na' in data bytes
- Rows with any 'na'
- Skipped rows (if applicable)

### 5. Testing

Comprehensive test suite includes:
- `test_safe_hex_to_int()` - Function behavior with both modes
- `test_split_payload_with_na()` - Payload processing
- `test_na_statistics()` - Statistics tracking
- `test_vocabulary_with_na_token()` - Vocab mapping
- `test_treat_na_as_modes()` - Different handling modes
- `test_na_handling_integration.py` - Full workflow integration

All tests passing ✅

### 6. Documentation

Updated `docs/PREPROCESSING.md` with:
- Explanation of why 'na' is important (attack signature)
- Usage examples for all three modes
- Technical details of vocabulary mapping
- Comparison of different strategies

## Why This Matters

### Before (Wrong) ❌
```
Attack with 'na':  can_id='na' → 0 → vocab[0] = "same as padding"
Normal CAN ID 0:   can_id=0x0  → 0 → vocab[0] = "same as attack"
```
Model cannot distinguish attacks from normal traffic!

### After (Correct) ✅
```
Attack with 'na':  can_id='na' → -1 → vocab[0] = "ATTACK_INDICATOR"
Normal CAN ID 0:   can_id=0x0  → 0  → vocab[1] = "normal_id_0"
```
Model learns that vocab[0] is a strong attack signal!

## Security Review

- ✅ CodeQL scan completed: No vulnerabilities found
- ✅ Code review completed: All issues addressed
- ✅ All tests passing
- ✅ Documentation updated

## Usage Examples

### Process with default (recommended) settings:
```bash
DATASET=set_02 python preprocessing/CAN_preprocess.py
```

### Process with old behavior (zero mode):
```bash
DATASET=set_02 python preprocessing/CAN_preprocess.py --treat-na-as zero
```

### Process and skip 'na' rows:
```bash
DATASET=set_02 python preprocessing/CAN_preprocess.py --treat-na-as skip
```

## Files Changed

1. `preprocessing/CAN_preprocess.py` - Core implementation
2. `tests/test_preprocessing.py` - Unit tests
3. `tests/test_preprocessing_basic.py` - Basic validation tests
4. `tests/test_na_handling_integration.py` - Integration tests
5. `docs/PREPROCESSING.md` - Documentation

## Backward Compatibility

- Default behavior is now `special_token` (preserves attack info)
- Old behavior available via `--treat-na-as zero`
- Existing scripts using `--skip-invalid-rows` still work

## Implementation Complete ✅

All requirements from the problem statement have been implemented and tested:
- ✅ Special token handling (-1 for 'na')
- ✅ Vocabulary mapping with reserved index 0
- ✅ Command-line arguments
- ✅ Statistics tracking with NAStatistics class
- ✅ Comprehensive tests
- ✅ Documentation updates
- ✅ Security scan (no issues)
- ✅ Code review (all issues addressed)
