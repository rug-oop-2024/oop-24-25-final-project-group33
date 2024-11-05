from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Detects feature types in the dataset.

    Args:
        dataset: Dataset object containing the data.

    Returns:
        List[Feature]: List of Feature objects with their types.
    """
    df = dataset.read()
    features = []

    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    for col in numerical_columns:
        features.append(Feature(name=col, type='numerical'))

    # Identify categorical columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns
    for col in categorical_columns:
        features.append(Feature(name=col, type='categorical'))

    return features
