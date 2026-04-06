# Quick Start Guide

## 1. Set API Key

```bash
export TOGETHER_API_KEY="your_api_key_here"
```

```powershell
$env:TOGETHER_API_KEY="your_api_key_here"
```

## 2. Quick Test (5 train episodes, 3 test episodes)

```bash
python scripts/run_p0_test.py --train-eps 5  --test-eps 3 --log-dir logs_quick --model-dir models_quick
```

## 3. Full Training and Testing (100 train episodes, 30 test episodes)

```bash
python scripts/run_p0_test.py --train-eps 100 --test-eps 30 --log-dir logs_exp1 --model-dir models_exp1
```


## 4. Test an Existing Model

Edit `MODEL_DIR` in `scripts/test_p0.py` to point to your saved checkpoint directory, then run:

```bash
python scripts/test_p0.py
```

## 5. MATH/svamp Dataset Test

```bash
python scripts/run_math_test.py --train-eps 12 --test-eps 5 --log-dir logs_math --model-dir models_math
``` 
