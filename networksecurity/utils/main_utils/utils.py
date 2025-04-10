import yaml
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV

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
        if replace and os.path.exists(filepath):
            os.remove(filepath)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e)


def save_numpy_array_data(filepath: str, array: np.array) -> None:
    """
    Saves a numpy array to a file.

    Args:
        filepath (str): Path to the file where the array will be saved.
        array (np.array): Numpy array to save.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e)


def save_object(filepath: str, obj: object) -> None:
    """
    Saves an object to a file using pickle.

    Args:
        filepath (str): Path to the file where the object will be saved.
        obj (object): Object to save.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e)


def load_numpy_array_data(filepath: str) -> np.ndarray:
    """
    Loads a numpy array from a file.

    Args:
        filepath (str): Path to the file from which the array will be loaded.

    Returns:
        np.array: Loaded numpy array.
    """
    try:
        with open(filepath, "rb") as file_obj:
            array = np.load(file_obj)
        return array
    except Exception as e:
        raise NetworkSecurityException(e)


def load_object(filepath: str) -> object:
    """
    Loads an object from a file using pickle.

    Args:
        filepath (str): Path to the file from which the object will be loaded.

    Returns:
        object: Loaded object.
    """
    try:
        with open(filepath, "rb") as file_obj:
            obj = pickle.load(file_obj)
        return obj
    except Exception as e:
        raise NetworkSecurityException(e)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
    params: dict,
) -> dict:
    """
    Evaluates multiple machine learning models and returns a report of their performance
    and the best trained model based on R² score.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        models (dict): Dictionary of model names and their corresponding instances.
        params (dict): Dictionary of model names and their corresponding hyperparameters.

    Returns:
        dict: A dictionary containing the evaluation report, best model name, best model instance,
              best score, and best parameters.
    """
    try:
        report = {}
        best_score = float("-inf")
        best_model = None
        best_model_name = None
        best_params = None

        for model_name, model in models.items():
            # Getting possible model parameters from the params dictionary
            model_params = params.get(model_name, {})

            # Performing Grid Search for hyperparameter tuning
            gridcv = GridSearchCV(model, model_params, cv=3)
            gridcv.fit(X_train, y_train)

            # Predicting and evaluating the model
            best_estimator = gridcv.best_estimator_
            y_preds = best_estimator.predict(X_test)
            score = r2_score(y_test, y_preds)

            report[model_name] = score

            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model = best_estimator
                best_params = gridcv.best_params_

        return {
            "report": report,
            "best_model_name": best_model_name,
            "best_model": best_model,
            "best_score": best_score,
            "best_params": best_params,
        }

    except Exception as e:
        raise NetworkSecurityException(e)
