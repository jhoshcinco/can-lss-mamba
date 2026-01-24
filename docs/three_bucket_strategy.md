# Three-Bucket Strategy: Avoiding Data Leakage in ML Research

## Overview

The **Three-Bucket Strategy** is a fundamental principle in machine learning research to ensure unbiased evaluation and prevent data leakage. This document explains how we implement this strategy in CAN-LSS-Mamba.

## What is Data Leakage?

**Data leakage** occurs when information from the test set influences model training or hyperparameter selection. This leads to:
- ❌ Overly optimistic performance estimates
- ❌ Poor generalization to real-world data
- ❌ Invalid research conclusions
- ❌ Wasted time on models that don't actually work

## The Three Buckets

### Bucket 1: Training Data (80%)
**Purpose**: Learn model parameters (weights, biases)
**Source**: 80% of `train_02_with_attacks/` folder
**Used by**: `train.py` for gradient descent optimization

### Bucket 2: Validation Data (20%)
**Purpose**: Model selection and hyperparameter tuning
**Source**: 20% of `train_02_with_attacks/` folder  
**Used by**: `train.py` for early stopping and threshold selection

**What you CAN do with Bucket 2**:
✅ Select best epoch (early stopping)
✅ Tune learning rate, batch size, epochs
✅ Select optimal classification threshold
✅ Choose between model architectures
✅ Run grid search / hyperparameter optimization

**What you CANNOT do with Bucket 2**:
❌ Final performance reporting (use Bucket 3)
❌ Claim generalization (test on Bucket 3)

### Bucket 3: Test Data
**Purpose**: Final unbiased evaluation
**Source**: `test_01/`, `test_02/`, ..., `test_06/` folders
**Used by**: `evaluate.py` for final performance reporting

**Critical Rules**:
⚠️ **NEVER** use test metrics for hyperparameter tuning
⚠️ **NEVER** re-train based on test results
⚠️ **NEVER** look at test data until hyperparameters are finalized
⚠️ Use test metrics for final thesis/paper results ONLY

## Checklist: Am I Avoiding Data Leakage?

Use this checklist before finalizing your research:

- [ ] I tuned hyperparameters using **validation** metrics (Bucket 2)
- [ ] I did NOT look at test results until hyperparameters were finalized
- [ ] I did NOT re-train after seeing test results
- [ ] I did NOT adjust thresholds based on test metrics
- [ ] I ran evaluation on test data (Bucket 3) **exactly once**
- [ ] My thesis/paper reports test metrics, not validation metrics
- [ ] I can explain why my validation/test performance differ (if they do)

If you checked all boxes, congratulations! Your research is sound. 🎉
