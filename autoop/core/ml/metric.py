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
    """
    Factory function to get a metric by name.
    Args:
        name (str): The name of the metric.
    Returns:
        Any: An instance of the requested metric.
    Raises:
        ValueError: If the metric name is not recognized.
    """
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
    """
    Base class for all metrics.
    Methods:
        __call__(y_true, y_pred): Calculate the metric value.
        evaluate(y_true, y_pred): Alias for the __call__ method.
    Args:
        y_true (Any): True labels.
        y_pred (Any): Predicted labels.
    Returns:
        float: Metric value.
    """

    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any) -> float:
        """
        Calculate the metric value.
        Args:
            y_true (Any): True labels.
            y_pred (Any): Predicted labels.
        Returns:
            float: Metric value.
        """
        pass

    def evaluate(self, y_true: Any, y_pred: Any) -> float:
        """
        Evaluate predictions.
        Args:
            y_true (Any): True labels.
            y_pred (Any): Predicted labels.
        Returns:
            float: Evaluation score.
        """
        return self.__call__(y_true, y_pred)


# Regression Metrics
class MeanSquaredError(Metric):
    """
    Mean Squared Error metric.
    Calculates the mean squared error between y_true and y_pred.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: Mean squared error.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate mean squared error.
        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: Mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error metric.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: Mean absolute error.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate mean absolute error.
        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: Mean absolute error.
        """
        return np.mean(np.abs(y_true - y_pred))


class RSquared(Metric):
    """
    R-squared metric (coefficient of determination).
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: R-squared value.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R-squared value.
        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: R-squared value.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


# Classification Metrics
class Accuracy(Metric):
    """
    Accuracy metric.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Accuracy score.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: Accuracy score.
        """
        return np.sum(y_true == y_pred) / len(y_true)


class Precision(Metric):
    """
    Precision metric for multi-class classification.
    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
    Returns:
        float: Calculated precision.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision for each class and return the average.
        Args:
            y_true (np.ndarray): True class labels.
            y_pred (np.ndarray): Predicted class labels.
        Returns:
            float: Average precision across all classes.
        """
        classes = np.unique(y_true)
        precision_sum = 0
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            precision_sum += tp / (tp + fp + 1e-10)  # Avoid division by zero
        return precision_sum / len(classes)


class Recall(Metric):
    """
    Recall metric for multi-class classification.
    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
    Returns:
        float: Calculated recall value.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall for each class and return the average.
        Args:
            y_true (np.ndarray): True class labels.
            y_pred (np.ndarray): Predicted class labels.
        Returns:
            float: Average recall across all classes.
        """
        classes = np.unique(y_true)
        recall_sum = 0
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall_sum += tp / (tp + fn + 1e-10)  # Avoid division by zero
        return recall_sum / len(classes)
