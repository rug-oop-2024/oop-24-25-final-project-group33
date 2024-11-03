from pydantic import BaseModel, Field
from typing import Optional
import base64

class Artifact(BaseModel):
    name: Optional[str] = Field(None, description="The name of the artifact.")
    type: Optional[str] = Field(None, description="The type of the artifact.")
    data: Optional[bytes] = Field(None, description="The binary data of the artifact.")

    def encode_data(self, data: str) -> bytes:
        """
        Encodes a string as base64 bytes and stores it in `self.data`.
        """
        encoded_data = base64.b64encode(data.encode())
        self.data = encoded_data
        return encoded_data

    def decode_data(self) -> str:
        """
        Decodes the base64 bytes in `self.data` and returns it as a string.
        """
        if self.data is None:
            raise ValueError("No data to decode.")
        return base64.b64decode(self.data).decode()

    def save(self, data: bytes):
        """
        Save raw bytes to `self.data`.
        """
        self.data = data

    def read(self) -> bytes:
        """
        Returns raw bytes from `self.data`.
        """
        if self.data is None:
            raise ValueError("No data to read.")
        return self.data
