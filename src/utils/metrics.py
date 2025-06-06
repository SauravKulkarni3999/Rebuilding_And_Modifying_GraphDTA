# -*- coding: utf-8 -*-
"""metrics

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ID1KESHk1JiBlu4VlkSgG188MyJODOAj
"""

# src/utils/metrics.py
import torch
import numpy as np # Retained for potential use, though current functions are pure PyTorch

def rmse_torch(y_pred, y_true):
    """
    Calculates Root Mean Squared Error using PyTorch tensors.
    """
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float)

    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()

    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

def concordance_index_torch(y_true, y_pred):
    """
    Calculates Concordance Index using PyTorch tensors or lists.
    Assumes y_true and y_pred are 1D.
    """
    if not isinstance(y_true, list):
        y_true = y_true.squeeze().tolist()
    if not isinstance(y_pred, list):
        y_pred = y_pred.squeeze().tolist()

    n = len(y_true)
    if n < 2:
        return 0.0  # Not enough pairs to compare

    concordant_pairs = 0
    permissible_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]: # Only consider pairs with different true values
                permissible_pairs += 1
                # Check for concordance
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant_pairs += 1
                # Check for ties in prediction for different true values (count as 0.5)
                elif y_pred[i] == y_pred[j]:
                    concordant_pairs += 0.5

    return concordant_pairs / permissible_pairs if permissible_pairs > 0 else 0.0


if __name__ == '__main__':
    # Example Usage
    true_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_values_good = torch.tensor([1.1, 2.2, 2.9, 4.1, 4.8])
    pred_values_bad = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    pred_values_mixed = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])

    print(f"RMSE (good): {rmse_torch(pred_values_good, true_values):.4f}")
    print(f"CI (good): {concordance_index_torch(true_values, pred_values_good):.4f}")

    print(f"RMSE (bad): {rmse_torch(pred_values_bad, true_values):.4f}")
    print(f"CI (bad): {concordance_index_torch(true_values, pred_values_bad):.4f}")

    print(f"RMSE (mixed): {rmse_torch(pred_values_mixed, true_values):.4f}")
    print(f"CI (mixed): {concordance_index_torch(true_values, pred_values_mixed):.4f}")

    # Test with lists
    true_list = [1, 2, 3, 2, 4]
    pred_list = [0.5, 1.5, 2.5, 3.5, 3] # Pair (1,2) vs (3,4) -> true[3]>true[0], pred[3]>pred[0]
    print(f"CI (list): {concordance_index_torch(true_list, pred_list):.4f}")