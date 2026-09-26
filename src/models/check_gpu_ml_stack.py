from __future__ import annotations

import sys

import xgboost
import catboost

print("Python:", sys.version)
print("XGBoost:", xgboost.__version__)
print("CatBoost:", catboost.__version__)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA runtime:", torch.version.cuda)
except ImportError:
    print("PyTorch: not installed (not required for the GBDT pipeline)")
