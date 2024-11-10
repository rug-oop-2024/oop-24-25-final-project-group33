from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class representing a dataset artifact.
    Attributes:
        None
    Methods:
        __init__(*args, **kwargs): Initialize the dataset.
        from_dataframe(data, name, asset_path, version="1.0.0"):
        Create a dataset from a pandas DataFrame.
        read() -> pd.DataFrame: Read the dataset as a pandas DataFrame.
        save(data) -> bytes: Save the dataset to the storage.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the dataset with given arguments.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ):
        """
        Create a Dataset from a DataFrame.
        Args:
            data (pd.DataFrame): The input data.
            name (str): The dataset name.
            asset_path (str): Path to save the dataset.
            version (str, optional): Dataset version. Defaults to "1.0.0".
        Returns:
            Dataset: The created dataset.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset and returns it as a pandas DataFrame.
        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save the dataset to storage.
        Args:
            data (pd.DataFrame): The dataset to save.
        Returns:
            bytes: The saved dataset in bytes.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
