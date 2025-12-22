"""
Evaluation Metrics for Wakeword Detection
"""

import numpy as np
import torch
from typing import Union, Optional

try:
    from sklearn.metrics import auc, roc_curve
    _HAS_SK = True
except ImportError:
    _HAS_SK = False

def _probs_from_logits(logits: Union[torch.Tensor, np.ndarray], positive_index: int = 1) -> np.ndarray:
    """
    Converts logits to positive class probabilities.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    
    if logits.ndim == 2 and logits.shape[1] == 2:
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[:, positive_index]
    elif logits.ndim == 1:
        # Assume these are already probabilities
        return logits
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

def calculate_pauc(
    logits: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    fpr_max: float = 0.1,
    positive_index: int = 1
) -> float:
    """
    Calculate the Partial Area Under the ROC Curve (pAUC) up to a max FPR.
    
    Args:
        logits: Model outputs (N, 2) or probabilities (N,)
        labels: Ground truth labels (N,)
        fpr_max: Maximum False Positive Rate to integrate up to.
        positive_index: Index of the positive class.
        
    Returns:
        pAUC value normalized by fpr_max (0.0 to 1.0)
    """
    y_true = labels
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    probs = _probs_from_logits(logits, positive_index)
    
    if _HAS_SK:
        fpr, tpr, _ = roc_curve(y_true, probs, drop_intermediate=False)
        
        # Filter FPR/TPR up to fpr_max
        mask = fpr <= fpr_max
        fpr_filtered = fpr[mask]
        tpr_filtered = tpr[mask]
        
        if len(fpr_filtered) < 2:
            return 0.0
            
        # Add a point at (fpr_max, interp(tpr)) to ensure we cover the full range
        if fpr_filtered[-1] < fpr_max:
            tpr_at_max = np.interp(fpr_max, fpr, tpr)
            fpr_filtered = np.append(fpr_filtered, fpr_max)
            tpr_filtered = np.append(tpr_filtered, tpr_at_max)
            
        area = auc(fpr_filtered, tpr_filtered)
        return float(area / fpr_max)
    else:
        # Fallback implementation without sklearn
        thresholds = np.sort(np.unique(np.concatenate(([0.0, 1.0], probs))))[::-1]
        fpr_list = []
        tpr_list = []
        
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            tn = np.sum((preds == 0) & (y_true == 0))
            
            curr_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            curr_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if curr_fpr > fpr_max:
                # Interpolate TPR at fpr_max
                if len(fpr_list) > 0:
                    prev_fpr = fpr_list[-1]
                    prev_tpr = tpr_list[-1]
                    alpha = (fpr_max - prev_fpr) / (curr_fpr - prev_fpr)
                    interp_tpr = prev_tpr + alpha * (curr_tpr - prev_tpr)
                    fpr_list.append(fpr_max)
                    tpr_list.append(interp_tpr)
                break
                
            fpr_list.append(curr_fpr)
            tpr_list.append(curr_tpr)
            
        if len(fpr_list) < 2:
            return 0.0
            
        area = np.trapz(tpr_list, fpr_list)
        return float(area / fpr_max)
