from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        """ Initialize the dataset. """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ):
        """ Create a dataset from a pandas DataFrame. """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """ Read the dataset as a pandas DataFrame. """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """ Save the dataset to the storage. """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
