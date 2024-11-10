from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Exception raised when a specified path cannot be found."""

    def __init__(self, path: str) -> None:
        """
        Initialize the NotFoundError with the missing path.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class defining a storage interface.
    Methods:
        save(data: bytes, path: str) -> None:
        load(path: str) -> bytes:
        delete(path: str) -> None:
        list(path: str) -> List[str]:
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a specified path.

        Args:
            data (bytes): Data to be saved.
            path (str): Path where the data will be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a specified path.

        Args:
            path (str): Path from which to load data.

        Returns:
            bytes: The loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a specified path.

        Args:
            path (str): Path of the data to delete.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all files under a specified path.

        Args:
            path (str): Path to list files under.

        Returns:
            List[str]: A list of file paths.
        """
        pass


class LocalStorage(Storage):
    """A local storage system using the file system for data persistence."""

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize the local storage system with a base path.

        Args:
            base_path (str): The base directory for storage.
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to the storage under the specified key.

        Args:
            data (bytes): Data to be saved.
            key (str): Key or path to store the data under.
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from the storage using the specified key.

        Args:
            key (str): Key or path to load data from.

        Returns:
            bytes: The loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data from the storage at the specified key.

        Args:
            key (str): Key or path to delete data from.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all files under the specified prefix path.

        Args:
            prefix (str): Prefix path to list files under.

        Returns:
            List[str]: List of relative file paths under the prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [
            os.path.relpath(
                p, self._base_path) for p in keys if os.path.isfile(p)
        ]

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a path exists and raise NotFoundError if it doesn't.

        Args:
            path (str): Path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Combine a relative path with the base path.

        Args:
            path (str): Path to combine with the base path.

        Returns:
            str: The resulting full path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
