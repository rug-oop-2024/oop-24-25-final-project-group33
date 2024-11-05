from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification import (
    LogisticRegressionModel,
    DecisionTreeClassifierModel,
    RandomForestClassifierModel,
)
from autoop.core.ml.model.regression import (
    LinearRegressionModel,
    RidgeRegressionModel,
    DecisionTreeRegressorModel,
)

REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Decision Tree Regressor",
]

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "Logistic Regression":
        return LogisticRegressionModel()
    elif model_name == "Decision Tree Classifier":
        return DecisionTreeClassifierModel()
    elif model_name == "Random Forest Classifier":
        return RandomForestClassifierModel()
    elif model_name == "Linear Regression":
        return LinearRegressionModel()
    elif model_name == "Ridge Regression":
        return RidgeRegressionModel()
    elif model_name == "Decision Tree Regressor":
        return DecisionTreeRegressorModel()
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")
