# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GSf64ABaQHsOdqYkrx8QtR7UFOoYM-Xw
"""

# src/utils/io_utils.py
import json
import os
import torch
import random
import numpy as np

def save_json_metrics(metrics: dict, output_path: str):
    """
    Saves the evaluation metrics to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_path}")

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed) # For numpy operations if any
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Potentially add torch.backends.cudnn settings if needed,
    # but they were in your report's experimental setup.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

if __name__ == '__main__':
    set_seed(123)
    example_metrics = {"RMSE": 0.5, "CI": 0.8}
    save_json_metrics(example_metrics, "dummy_metrics.json")
    if os.path.exists("dummy_metrics.json"):
        print("Dummy metrics file created successfully.")
        os.remove("dummy_metrics.json")