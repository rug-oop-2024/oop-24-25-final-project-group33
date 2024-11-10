# autoop/core/ml/model/regression.py
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


class MultipleLinearRegression(Model):
    """
    MultipleLinearRegression model.
    Methods:
        __init__(): Initialize the model.
        fit(X: np.ndarray, y: np.ndarray) -> None: Fit model to data.
        predict(X: np.ndarray) -> np.ndarray: Predict target variable.
    Attributes:
        model: Linear regression model.
    """

    def __init__(self) -> None:
        """
        Initializes a regression model instance.
        """
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model.
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        Returns:
            None
        """
        self.model.fit(X, y)
        self.set_params(**self.model.get_params())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the regression model.
        Args:
            X (np.ndarray): Input features.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


class LinearRegressionModel(Model):
    """
    LinearRegressionModel for regression tasks.
    Attributes:
        model (LinearRegression): The linear regression model.
        parameters (dict): Model parameters.
    Methods:
        __init__(): Initialize the model.
        fit(X, y): Fit the model to the training data.
        predict(X): Predict the target variable for the input data.
    """
    def __init__(self) -> None:
        """
        Initializes the regression model.
        """
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model.
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
            X (np.ndarray): Input features.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


class RidgeRegressionModel(Model):
    """
    RidgeRegressionModel class for Ridge regression.
    Methods:
        __init__(): Initialize the model.
        fit(X: np.ndarray, y: np.ndarray) -> None: Fit model to data.
        predict(X: np.ndarray) -> np.ndarray: Predict target variable.
    """
    def __init__(self) -> None:
        """
        Initializes the regression model with Ridge regression.
        """
        super().__init__(model_type="regression")
        self.model = Ridge()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        Returns:
            None
        """
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.
        Args:
            X (np.ndarray): Input features.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)


class DecisionTreeRegressorModel(Model):
    """
    DecisionTreeRegressorModel: A model for regression using decision trees.
    Methods:
        __init__: Initialize the model.
        fit: Fit the model to the training data.
            Args:
                X (np.ndarray): Training data features.
                y (np.ndarray): Training data target.
            Returns:
                None
        predict: Predict the target variable for the input data.
            Args:
                X (np.ndarray): Input data features.
            Returns:
                np.ndarray: Predicted target variable.
    """
    def __init__(self) -> None:
        """
        Initialize regression model with DecisionTreeRegressor.
        """
        super().__init__(model_type="regression")
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model.
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
        Predict using the regression model.
        Args:
            X (np.ndarray): Input features.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)
