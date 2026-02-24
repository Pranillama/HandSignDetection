"""Model evaluation metrics and performance tracking utilities."""

from __future__ import annotations

from typing import List, Dict, Union, Optional
import warnings

import numpy as np


def calculate_accuracy(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute overall accuracy between true and predicted labels.

    Args:
        y_true: Array-like of integer ground-truth class labels.
        y_pred: Array-like of integer predicted class labels.

    Returns:
        Float accuracy (fraction correct).

    Raises:
        ValueError: if `y_true` and `y_pred` have different lengths.
    """
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    if arr_true.shape != arr_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(arr_true == arr_pred))


def confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    num_classes: int,
) -> np.ndarray:
    """Build a confusion matrix.

    Rows correspond to true labels and columns to predicted labels.

    Args:
        y_true: Array-like of integer ground-truth class labels.
        y_pred: Array-like of integer predicted class labels.
        num_classes: Total number of classes.

    Returns:
        An integer NumPy array of shape (num_classes, num_classes).

    Raises:
        ValueError: if labels are outside the [0, num_classes) range or
            if inputs have mismatched lengths.
    """
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    if arr_true.shape != arr_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(arr_true, arr_pred):
        if t < 0 or t >= num_classes or p < 0 or p >= num_classes:
            raise ValueError(
                f"Labels must be in [0, {num_classes}); received true={t}, pred={p}"
            )
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: List[str],
) -> Dict[str, Optional[float]]:
    """Compute accuracy for each class.

    Args:
        y_true: Array-like of integer ground-truth class labels.
        y_pred: Array-like of integer predicted class labels.
        class_names: List of class names in order of label index.

    Returns:
        Dictionary mapping each class name to its accuracy. If a class
        has no samples, its value is `None` and a warning is emitted.
    """
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    num_classes = len(class_names)
    if arr_true.shape != arr_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    results: Dict[str, Optional[float]] = {}
    for idx, name in enumerate(class_names):
        mask = arr_true == idx
        total = int(mask.sum())
        if total == 0:
            warnings.warn(f"No samples found for class '{name}' (index {idx})")
            results[name] = None
        else:
            correct = int((arr_pred[mask] == idx).sum())
            results[name] = correct / total
    return results


def calculate_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: List[str],
) -> Dict[str, Union[float, list, Dict[str, Optional[float]]]]:
    """Aggregate multiple evaluation metrics into a single dictionary.

    This helper constructs the structure expected by the project's
    `metrics.json` schema.

    Args:
        y_true: Array-like of ground-truth labels.
        y_pred: Array-like of predicted labels.
        class_names: Names of classes in label order.

    Returns:
        A dictionary with keys ``accuracy``, ``confusion_matrix`` (as a
        nested list), and ``per_class_accuracy``.
    """
    acc = calculate_accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, len(class_names)).tolist()
    per_cls = per_class_accuracy(y_true, y_pred, class_names)
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "per_class_accuracy": per_cls,
    }
