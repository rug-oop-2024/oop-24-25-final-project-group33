from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class representing a dataset artifact that can \
be read from and saved as a CSV.

    Inherits from:
        Artifact: Base artifact class.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the dataset artifact with a specified type.

        Args:
            *args: Positional arguments for the Artifact base class.
            **kwargs: Keyword arguments for the Artifact base class.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ) -> "Dataset":
        """
        Create a Dataset instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The data to be saved in the dataset.
            name (str): Name of the dataset.
            asset_path (str): Path to the asset in storage.
            version (str, optional): Version of the dataset. \
Defaults to "1.0.0".

        Returns:
            Dataset: An instance of the Dataset class.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Read and return the dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset loaded into a DataFrame.
        """
        bytes_data = super().read()
        csv_data = bytes_data.decode()
        return pd.read_csv(io.StringIO(csv_data))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save the dataset from a pandas DataFrame to storage.

        Args:
            data (pd.DataFrame): The DataFrame to save as CSV in bytes.

        Returns:
            bytes: The CSV data in byte format after saving.
        """
        bytes_data = data.to_csv(index=False).encode()
        return super().save(bytes_data)
