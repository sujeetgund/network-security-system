import os
from scipy.stats import ks_2samp
import pandas as pd

from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.utils.main_utils import read_yaml, write_yaml
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger


class DataValidation:
    def __init__(
        self,
        config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.config = config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads the data from the given file path.

        Args:
            file_path (str): Path to the data file.

        Raises:
            FileNotFoundError: If the file does not exist.
            NetworkSecurityException: If there is an error reading the file.

        Returns:
            pd.DataFrame: DataFrame containing the data.
        """
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise NetworkSecurityException(e)

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        """
        Validates the number of columns in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: True if the number of columns is valid, False otherwise.
        """
        try:
            print(self.schema_config)
            expected_columns = self.schema_config["number_of_columns"]
            actual_columns = df.shape[1]

            logger.info(f"Expected number of columns: {expected_columns}")
            logger.info(f"Actual number of columns: {actual_columns}")

            return actual_columns == expected_columns
        except Exception as e:
            raise NetworkSecurityException(e)

    def detect_data_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """
        Detects data drift between the base DataFrame and the current DataFrame.

        Args:
            base_df (pd.DataFrame): Base DataFrame.
            current_df (pd.DataFrame): Current DataFrame.
            threshold (float): p-value threshold for drift detection.
                Default is 0.05.

        Returns:
            bool: True if drift is detected, False otherwise.
        """
        try:
            report = {}
            status = False
            for column in base_df.columns:
                test = ks_2samp(base_df[column], current_df[column])
                p_value = float(test.pvalue)
                drift_detected = p_value > threshold
                if drift_detected:
                    status = True

                report.update(
                    {
                        column: {
                            "drift_detected": drift_detected,
                            "p_value": p_value,
                        }
                    }
                )

            drift_report_path = self.config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            write_yaml(drift_report_path, report)
            logger.info(f"Drift report saved at {drift_report_path}")

            return status
        except Exception as e:
            raise NetworkSecurityException(e)

    def init_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            logger.info("Validating number of columns in train and test data")
            # Validate number of columns
            if not self.validate_number_of_columns(train_df):
                raise ValueError("Number of columns in the training data is invalid.")
            if not self.validate_number_of_columns(test_df):
                raise ValueError("Number of columns in the testing data is invalid.")

            logger.info("Checking for data drift")
            # Check for data drift
            drift_status = self.detect_data_drift(train_df, test_df)

            logger.info("Saving valid train and test data")
            # Save valid train and test data
            valid_train_file_path = self.config.valid_train_file_path
            valid_test_file_path = self.config.valid_test_file_path

            os.makedirs(os.path.dirname(valid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(valid_test_file_path), exist_ok=True)

            train_df.to_csv(valid_train_file_path, index=False)
            test_df.to_csv(valid_test_file_path, index=False)

            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.config.drift_report_file_path,
            )

            logger.info("Data validation completed successfully")

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e)
