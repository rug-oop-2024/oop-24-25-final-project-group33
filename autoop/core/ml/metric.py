from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# List of available metrics
METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared",
    "accuracy",
    "precision",
    "recall",
]


def get_metric(name: str) -> Any:
    """Factory function to get a metric by name."""
    metrics = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "r_squared": RSquared(),
        "accuracy": Accuracy(),
        "precision": Precision(),
        "recall": Recall(),
    }
    if name in metrics:
        return metrics[name]
    else:
        raise ValueError(f"Metric '{name}' is not recognized. \
Available metrics are: {list(metrics.keys())}")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any) -> float:
        """Calculate the metric value."""
        pass

    def evaluate(self, y_true: Any, y_pred: Any) -> float:
        """Alias for the __call__ method."""
        return self.__call__(y_true, y_pred)


# Regression Metrics
class MeanSquaredError(Metric):
    """Mean Squared Error metric."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the mean squared error."""
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the mean absolute"""
        return np.mean(np.abs(y_true - y_pred))


class RSquared(Metric):
    """R-squared metric (coefficient of determination)."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the R-squared value."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


# Classification Metrics
class Accuracy(Metric):
    """Accuracy metric."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the accuracy."""
        return np.sum(y_true == y_pred) / len(y_true)


class Precision(Metric):
    """Precision metric for multi-class classification."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the precision."""
        classes = np.unique(y_true)
        precision_sum = 0
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            precision_sum += tp / (tp + fp + 1e-10)  # Avoid division by zero
        return precision_sum / len(classes)


class Recall(Metric):
    """Recall metric for multi-class classification."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the recall."""
        classes = np.unique(y_true)
        recall_sum = 0
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall_sum += tp / (tp + fn + 1e-10)  # Avoid division by zero
        return recall_sum / len(classes)
