# autoop/core/ml/model/classification.py
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class LogisticRegressionModel(Model):
    """
    LogisticRegressionModel for classification tasks.
    Attributes:
        model (LogisticRegression): The logistic regression model.
        parameters (dict): Parameters of the logistic regression model.
    Methods:
        __init__(): Initialize the model.
        fit(X: np.ndarray, y: np.ndarray): Fit the model to the training data.
        predict(X: np.ndarray): Predict the target variable for the input data.
    """
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="classification")
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        Returns:
            None
        """
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)


class DecisionTreeClassifierModel(Model):
    """
    DecisionTreeClassifierModel for classification tasks.
    Attributes:
        model (DecisionTreeClassifier): The decision tree classifier.
        parameters (dict): Parameters of the decision tree model.
    Methods:
        __init__(): Initialize the model.
        fit(X, y): Fit the model to the training data.
        predict(X): Predict the target variable for the input data.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target variable.
    Returns:
        np.ndarray: Predicted target variable.
    """
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="classification")
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        Returns:
            None
        """
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


class RandomForestClassifierModel(Model):
    """
    RandomForestClassifierModel for classification tasks.
    Attributes:
        model (RandomForestClassifier): The RandomForestClassifier instance.
        parameters (dict): Parameters of the fitted model.
    Methods:
        __init__(): Initialize the model.
        fit(X, y): Fit the model to the training data.
            Args:
                X (np.ndarray): Training data features.
                y (np.ndarray): Training data labels.
            Returns:
                None
        predict(X): Predict the target variable for the input data.
            Args:
                X (np.ndarray): Input data features.
            Returns:
                np.ndarray: Predicted labels.
    """
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="classification")
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        Returns:
            None
        """
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)
