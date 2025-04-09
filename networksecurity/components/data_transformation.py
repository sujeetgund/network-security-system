import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from networksecurity.constant.training_pipeline import (
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)

from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger
from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(
        self,
        config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.config = config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e)

    @staticmethod
    def read_data(filepath: str) -> pd.DataFrame:
        """
        Reads the data from the given filepath.

        Args:
            filepath (str): Path to the data file

        Raises:
            NetworkSecurityException: If there is an error in reading the file

        Returns:
            pd.DataFrame: DataFrame containing the data
        """
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise NetworkSecurityException(e)

    def get_transformation_object(self) -> Pipeline:
        """
        This method initializes the KNNImputer and returns a pipeline object.
        The KNNImputer is used to fill missing values in the dataset.

        Raises:
            NetworkSecurityException: If there is an error in initializing the imputer

        Returns:
            Pipeline: A pipeline object containing the KNNImputer
        """
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            preprocessor = Pipeline(
                steps=[
                    ("imputer", imputer),
                ]
            )

            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e)

    def init_data_transformation(self) -> DataTransformationArtifact:
        """
        This method performs data transformation on the training and testing datasets.
        It reads the data, applies the KNNImputer to fill missing values, and saves the transformed data and preprocessor object.

        Raises:
            NetworkSecurityException: If there is an error in the data transformation process

        Returns:
            DataTransformationArtifact: An object containing the file paths of the transformed data and preprocessor object
        """
        logger.info("Data transformation started")
        try:
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Training
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            # Lets replace the -1 values with 0 and keep 1 as it is
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)

            # Testing
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            # Lets replace the -1 values with 0 and keep 1 as it is
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            preprocessor = self.get_transformation_object()

            # Fit the preprocessor on the training data and transform both train and test data
            logger.info("Fitting and transforming data")
            transformed_input_feature_train_arr = preprocessor.fit_transform(
                input_feature_train_df
            )
            transformed_input_feature_test_arr = preprocessor.transform(
                input_feature_test_df
            )

            train_arr = np.c_[
                transformed_input_feature_train_arr, target_feature_train_arr
            ]
            test_arr = np.c_[
                transformed_input_feature_test_arr, target_feature_test_arr
            ]

            # Save the transformed data to the specified file paths
            logger.info("Saving transformed data")
            save_numpy_array_data(
                filepath=self.config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                filepath=self.config.transformed_test_file_path,
                array=test_arr,
            )

            # Save the preprocessor object
            logger.info("Saving transformation object")
            save_object(
                filepath=self.config.transformed_object_file_path,
                obj=preprocessor,
            )

            logger.info("Data transformation completed")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.config.transformed_object_file_path,
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path,
            )

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e)
