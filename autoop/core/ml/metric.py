from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score

# List of available metrics
METRICS = [
    "mean_squared_error",
    "accuracy",
]

def get_metric(name: str):
    """Factory function to get a metric by name.
    
    Args:
        name (str): The name of the metric.
        
    Returns:
        Metric: An instance of the requested metric.
        
    Raises:
        ValueError: If the metric name is not recognized.
    """
    metrics = {
        "mean_squared_error": MeanSquaredError(),
        "accuracy": Accuracy(),
    }
    if name in metrics:
        return metrics[name]
    else:
        raise ValueError(f"Metric '{name}' is not recognized. Available metrics are: {list(metrics.keys())}")

class Metric(ABC):
    """Base class for all metrics."""
    
    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any) -> float:
        """Compute the metric.
        
        Args:
            y_true (Any): Ground truth values.
            y_pred (Any): Predicted values.
            
        Returns:
            float: Computed metric value.
        """
        pass

class MeanSquaredError(Metric):
    """Mean Squared Error metric."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Mean Squared Error."""
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Mean Squared Error."""
        return self.__call__(y_true, y_pred)


class Accuracy(Metric):
    """Accuracy metric."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Accuracy.
        
        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
            
        Returns:
            float: Accuracy.
        """
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
