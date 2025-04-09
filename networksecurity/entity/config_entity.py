import os
from datetime import datetime

from networksecurity.constant import training_pipeline


class TrainingPipelineConfig:
    def __init__(self):
        timestamps = datetime.now().strftime("%Y%m%d%H%M%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACTS_NAME
        self.artifact_dir = os.path.join(training_pipeline.ARTIFACTS_NAME, timestamps)


class DataIngestionConfig:
    def __init__(self, config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            config.artifact_dir, training_pipeline.DATA_INGESTION_NAME
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_NAME,
            training_pipeline.FEATURE_STORE_FILE_NAME,
        )
        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_NAME,
            training_pipeline.TRAIN_DATA_FILE_NAME,
        )
        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_NAME,
            training_pipeline.TEST_DATA_FILE_NAME,
        )
        self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.test_size = training_pipeline.DATA_INGESTION_TEST_SIZE


class DataValidationConfig:
    def __init__(self, config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            config.artifact_dir, training_pipeline.DATA_VALIDATION_NAME
        )

        self.valid_data_dir = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_NAME
        )
        self.valid_train_file_path = os.path.join(
            self.valid_data_dir, training_pipeline.TRAIN_DATA_FILE_NAME
        )
        self.valid_test_file_path = os.path.join(
            self.valid_data_dir, training_pipeline.TEST_DATA_FILE_NAME
        )

        self.invalid_data_dir = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_NAME
        )
        self.invalid_train_file_path = os.path.join(
            self.invalid_data_dir, training_pipeline.TRAIN_DATA_FILE_NAME
        )
        self.invalid_test_file_path = os.path.join(
            self.invalid_data_dir, training_pipeline.TEST_DATA_FILE_NAME
        )

        self.drift_report_file_path = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )
