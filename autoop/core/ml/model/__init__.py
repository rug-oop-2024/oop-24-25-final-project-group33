""" Module to manage the creation of model instances. """
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
    """
    Factory function to retrieve a model instance by name.

    Args:
        model_name (str): The name of the model to retrieve.
                          Must be one of the names listed in \
`REGRESSION_MODELS` or `CLASSIFICATION_MODELS`.

    Returns:
        Model: An instance of the specified model.

    Raises:
        ValueError: If `model_name` does not match any known model.

    Available Models:
        - Classification:
            "Logistic Regression": Returns a LogisticRegressionModel instance.
            "Decision Tree Classifier": \
Returns a DecisionTreeClassifierModel instance.
            "Random Forest Classifier": \
Returns a RandomForestClassifierModel instance.

        - Regression:
            "Linear Regression": Returns a LinearRegressionModel instance.
            "Ridge Regression": Returns a RidgeRegressionModel instance.
            "Decision Tree Regressor": \
Returns a DecisionTreeRegressorModel instance.
    """
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
