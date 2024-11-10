from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Literal
from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """
    Abstract base class for machine learning models.
    Attributes:
        model_type (str): Type of the model ('classification' or 'regression').
        parameters (Dict[str, Any]): Model parameters.
    Methods:
        fit(X: np.ndarray, y: np.ndarray) -> None:
        predict(X: np.ndarray) -> np.ndarray:
            Predict the target variable for the input data.
        set_params(**params: Any) -> None:
            Set the parameters of the model.
        get_params() -> Dict[str, Any]:
            Get the parameters of the model.
        to_artifact(name: str) -> Artifact:
        save(path: str) -> None:
        load(path: str) -> None:
    """

    def __init__(self, model_type: Literal["classification", "regression"]):
        """
        Initialize the model with type and parameters.
        Args:
            model_type (Literal["classification", "regression"]):
            Type of model.
        Attributes:
            _model_type (str): The type of the model.
            _parameters (Dict[str, Any]): Parameters of the model.
        """
        self._model_type = model_type
        self._parameters: Dict[str, Any] = {}

    @property
    def model_type(self) -> str:
        """
        Returns the type of the model.
        Returns:
            str: The model type.
        """
        return self._model_type

    @model_type.setter
    def model_type(self, model_type: str) -> None:
        """
        Set the model type.
        Args:
            model_type (str): Either 'classification' or 'regression'.
        Raises:
            ValueError: If model_type is not 'classification' or 'regression'.
        """

        """ Set the model type. """
        if model_type not in ["classification", "regression"]:
            raise ValueError(
                "model_type must be either 'classification' or 'regression'")
        self._model_type = model_type

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get the model parameters.

        Returns:
            Dict[str, Any]: A deep copy of the model parameters.
        """
        return deepcopy(self._parameters)

    def set_params(self, **params: Any) -> None:
        """
        Set parameters for the model.
        Args:
            **params (Any): Parameters to update.
        Returns:
            None
        """
        self._parameters.update(params)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the training data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted values.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Returns a deep copy of model parameters.
        """
        return deepcopy(self._parameters)

    def to_artifact(self, name: str) -> Artifact:
        """
        Convert the model to an artifact.

        Args:
            name (str): The name of the artifact.

        Returns:
            Artifact: The artifact containing the model type and parameters.
        """
        data = {
            "model_type": self._model_type,
            "parameters": self._parameters,
        }
        return Artifact(name=name, data=data)

    def save(self, path: str) -> None:
        """
        Save model parameters to a file.

        Args:
            path (str): The file path to save the model parameters.

        Returns:
            None
        """
        with open(path, 'wb') as file:
            np.save(file, self._parameters)

    def load(self, path: str) -> None:
        """
        Load model parameters from a file.
        Args:
            path (str): Path to the file.
        Returns:
            None
        """
        with open(path, 'rb') as file:
            self._parameters = np.load(file, allow_pickle=True).item()
