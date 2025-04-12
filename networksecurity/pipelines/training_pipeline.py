import os

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.cloud import S3Sync
from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME

from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger


class TrainingPipeline:
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.s3_sync = S3Sync()
        self.training_bucket_name = TRAINING_BUCKET_NAME

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This function is responsible for starting the data ingestion process.

        Raises:
            NetworkSecurityException: If there is an error during the data ingestion process.

        Returns:
            DataIngestionArtifact: An artifact containing information about the data ingestion process.
        """
        try:
            data_ingestion_config = DataIngestionConfig(config=self.config)
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.init_data_ingestion()

            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        This function is responsible for starting the data validation process.
        It takes the data ingestion artifact as input and performs validation checks on the ingested data.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): An artifact containing information about the data ingestion process.

        Raises:
            NetworkSecurityException: If there is an error during the data validation process.

        Returns:
            DataValidationArtifact: An artifact containing information about the data validation process.
        """
        try:
            data_validation_config = DataValidationConfig(config=self.config)
            data_validation = DataValidation(
                config=data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            data_validation_artifact = data_validation.init_data_validation()

            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        This function is responsible for starting the data transformation process.
        It takes the data validation artifact as input and performs transformation operations on the validated data.

        Args:
            data_validation_artifact (DataValidationArtifact): An artifact containing information about the data validation process.

        Raises:
            NetworkSecurityException: If there is an error during the data transformation process.

        Returns:
            DataTransformationArtifact: An artifact containing information about the data transformation process.
        """
        try:
            data_transformation_config = DataTransformationConfig(config=self.config)
            data_transformation = DataTransformation(
                config=data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            data_transformation_artifact = (
                data_transformation.init_data_transformation()
            )

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        This function is responsible for starting the model training process.
        It takes the data transformation artifact as input and trains a machine learning model.

        Args:
            data_transformation_artifact (DataTransformationArtifact): An artifact containing information about the data transformation process.

        Raises:
            NetworkSecurityException: If there is an error during the model training process.

        Returns:
            ModelTrainerArtifact: An artifact containing information about the model training process.
        """
        try:
            model_trainer_config = ModelTrainerConfig(config=self.config)
            model_trainer = ModelTrainer(
                config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            model_trainer_artifact = model_trainer.init_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e)

    def sync_artifact_dir_to_s3(self) -> None:
        """
        This function is responsible for syncing the artifact directory to S3.

        Raises:
            NetworkSecurityException: If there is an error during syncing the artifact directory to S3.

        Returns:
            None
        """
        logger.info("Syncing artifact directory to S3")
        try:
            aws_bucket_url = (
                f"s3://{self.training_bucket_name}/{self.config.artifact_dir}"
            )
            self.s3_sync.sync_folder_to_s3(
                local_folder=self.config.artifact_dir, bucket_url=aws_bucket_url
            )

            logger.info("Artifact directory synced to S3 successfully")
        except Exception as e:
            raise NetworkSecurityException(e)

    def sync_saved_model_dir_to_s3(self) -> None:
        """
        This function is responsible for syncing the saved model directory to S3.

        Raises:
            NetworkSecurityException: If there is an error during syncing the saved model directory to S3.

        Returns:
            None
        """
        logger.info("Syncing saved model directory to S3")
        try:
            aws_bucket_url = (
                f"s3://{self.training_bucket_name}/{self.config.saved_model_dir}"
            )
            self.s3_sync.sync_folder_to_s3(
                local_folder=self.config.saved_model_dir, bucket_url=aws_bucket_url
            )

            logger.info("Saved model directory synced to S3 successfully")
        except Exception as e:
            raise NetworkSecurityException(e)

    def run_pipeline(
        self, sync_artifact: bool = True, sync_model: bool = True
    ) -> ModelTrainerArtifact:
        """
        This function is responsible for running the entire training pipeline.
        It orchestrates the execution of various components such as data ingestion,
        data validation, data transformation, and model training.
        It returns the final model trainer artifact.

        Args:
            sync_artifact (bool, optional): Flag to indicate whether to sync the artifact directory to S3. Default is True.
            sync_model (bool, optional): Flag to indicate whether to sync the saved model directory to S3. Default is True.

        Raises:
            NetworkSecurityException: If there is an error during executing the training pipeline.

        Returns:
            ModelTrainerArtifact: An artifact containing information about the model training process.
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            # Syncing artifact and saved model directories to S3
            if sync_artifact:
                self.sync_artifact_dir_to_s3()
            if sync_model:
                self.sync_saved_model_dir_to_s3()

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e)
