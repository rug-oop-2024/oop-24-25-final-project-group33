from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """ A class representing a feature in a dataset. """
    name: str = Field(..., description="The name of the feature.")
    type: Literal['numerical', 'categorical'] = Field(
        ..., description="The type of the feature, \
either 'numerical' or 'categorical'."
    )

    def __str__(self) -> str:
        """ Return a string representation of the feature. """
        return f"Feature(name={self.name}, type={self.type})"
