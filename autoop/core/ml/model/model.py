# autoop/core/ml/model/model.py
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Literal
from autoop.core.ml.artifact import Artifact

class Model(ABC):
    def __init__(self, model_type: Literal["classification", "regression"]):
        self.model_type = model_type
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> Dict[str, Any]:
        return deepcopy(self.parameters)

    def set_params(self, **params: Any):
        self.parameters.update(params)

    def to_artifact(self, name: str) -> Artifact:
        data = {
            "model_type": self.model_type,
            "parameters": self.parameters,
        }
        return Artifact(name=name, data=data)

    def save(self, path: str):
        with open(path, 'wb') as file:
            np.save(file, self.parameters)

    def load(self, path: str):
        with open(path, 'rb') as file:
            self.parameters = np.load(file, allow_pickle=True).item()
