import yaml
import os
import dill
import numpy as np
import pickle

from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger


def read_yaml(filepath: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.
    """
    try:
        with open(filepath, "r") as file:
            content = yaml.safe_load(file)
        return content
    except Exception as e:
        raise NetworkSecurityException(e)


def write_yaml(filepath: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.

    Args:
        filepath (str): Path to the YAML file.
        content (object): Dictionary to write to the file.
        replace (bool): If True, replaces the existing file. Defaults to False.

    Returns:
        None
    """
    try:
        if os.path.exists(filepath) and not replace:
            raise FileExistsError(f"File already exists: {filepath}")

        if replace and os.path.exists(filepath):
            os.remove(filepath)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e)
