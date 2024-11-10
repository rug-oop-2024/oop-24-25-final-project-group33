from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    A class representing a feature in a dataset.
    Attributes:
        name (str): The name of the feature.
        type (Literal['numerical', 'categorical']): The type of the feature.
    Methods:
        __str__() -> str: Return a string representation of the feature.
    """
    name: str = Field(..., description="The name of the feature.")
    type: Literal['numerical', 'categorical'] = Field(
        ..., description="The type of the feature, \
either 'numerical' or 'categorical'."
    )

    def __str__(self) -> str:
        """
        Return a string representation of the feature.
        Returns:
            str: Feature details as a string.
        """
        return f"Feature(name={self.name}, type={self.type})"
