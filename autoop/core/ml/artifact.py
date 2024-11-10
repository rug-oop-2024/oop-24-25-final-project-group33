from pydantic import BaseModel, Field
from typing import Optional
import base64
from typing import List, Dict, Any
import hashlib


class Artifact(BaseModel):
    """
    A class representing an artifact.
    Attributes:
        name (Optional[str]): The name of the artifact.
        type (Optional[str]): The type of the artifact.
        data (Optional[bytes]): The binary data of the artifact.
        asset_path (Optional[str]): The path to the asset.
        version (Optional[str]): The version of the artifact.
        tags (Optional[List[str]]): List of tags associated with the artifact.
        metadata (Optional[Dict[str, Any]]):
        Additional metadata for the artifact.
    Properties:
        id (str): Generates a unique ID for the artifact.
    Methods:
        encode_data(data: str) -> bytes: Encodes a string as base64 bytes.
        decode_data() -> str: Decodes the base64 bytes in `self.data`.
        save(data: bytes) -> None: Save raw bytes to `self.data`.
        read() -> bytes: Returns raw bytes from `self.data`.
    """
    name: Optional[str] = Field(None, description="The name of the artifact.")
    type: Optional[str] = Field(None, description="The type of the artifact.")
    data: Optional[bytes] = Field(
        None, description="The binary data of the artifact.")
    asset_path: Optional[str] = Field(
        None, description="The path to the asset.")
    version: Optional[str] = Field(
        None, description="The version of the artifact.")
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="List of tags associated with the artifact.")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the artifact.")

    @property
    def id(self) -> str:
        """
        Generates a unique ID for the artifact.
        Returns:
            str: Unique ID based on name and version.
        """
        unique_string = f"{self.name}:{self.version}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def encode_data(self, data: str) -> bytes:
        """
        Encode a string to base64 bytes.
        Args:
            data (str): The string to encode.
        Returns:
            bytes: The base64 encoded bytes.
        """
        encoded_data = base64.b64encode(data.encode())
        self.data = encoded_data
        return encoded_data

    def decode_data(self) -> str:
        """
        Decodes base64 encoded data to a string.
        Returns:
            str: The decoded string.
        Raises:
            ValueError: If no data to decode.
        """
        if self.data is None:
            raise ValueError("No data to decode.")
        return base64.b64decode(self.data).decode()

    def save(self, data: bytes) -> None:
        """
        Save data as bytes.
        Args:
            data (bytes): Data to be saved.
        Returns:
            None
        """
        self.data = data

    def read(self) -> bytes:
        """
        Reads and returns the stored data as bytes.
        Returns:
            bytes: The stored data.
        Raises:
            ValueError: If no data is available to read.
        """
        if self.data is None:
            raise ValueError("No data to read.")
        return self.data
