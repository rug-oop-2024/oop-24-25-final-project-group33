from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Literal
from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """ A class representing a machine learning model. """

    def __init__(self, model_type: Literal["classification", "regression"]):
        """ Initialize the model. """
        self._model_type = model_type
        self._parameters: Dict[str, Any] = {}

    @property
    def model_type(self) -> str:
        """ Get the model type. """
        return self._model_type

    @model_type.setter
    def model_type(self, model_type: str) -> None:
        """ Set the model type. """
        if model_type not in ["classification", "regression"]:
            raise ValueError(
                "model_type must be either 'classification' or 'regression'")
        self._model_type = model_type

    @property
    def parameters(self) -> Dict[str, Any]:
        """ Get the parameters of the model. """
        return deepcopy(self._parameters)

    def set_params(self, **params: Any) -> None:
        """ Set the parameters of the model. """
        self._parameters.update(params)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Fit the model to the training data. """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predict the target variable for the input data. """
        pass

    def get_params(self) -> Dict[str, Any]:
        """ Get the parameters of the model. """
        return deepcopy(self._parameters)

    def to_artifact(self, name: str) -> Artifact:
        """ Convert the model to an artifact. """
        data = {
            "model_type": self._model_type,
            "parameters": self._parameters,
        }
        return Artifact(name=name, data=data)

    def save(self, path: str) -> None:
        """ Save the model to a file. """
        with open(path, 'wb') as file:
            np.save(file, self._parameters)

    def load(self, path: str) -> None:
        """ Load the model from a file. """
        with open(path, 'rb') as file:
            self._parameters = np.load(file, allow_pickle=True).item()
