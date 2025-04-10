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
        # Data Ingestion Dir
        self.data_ingestion_dir = os.path.join(
            config.artifact_dir, training_pipeline.DATA_INGESTION_NAME
        )

        # Feature Store File Path
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR_NAME,
            training_pipeline.FEATURE_STORE_FILE_NAME,
        )

        # Training File Path
        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME,
            training_pipeline.TRAIN_DATA_FILE_NAME,
        )

        # Testing File Path
        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME,
            training_pipeline.TEST_DATA_FILE_NAME,
        )

        # Database Name and Collection Name
        self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # Testing Size
        self.test_size = training_pipeline.DATA_INGESTION_TEST_SIZE


class DataValidationConfig:
    def __init__(self, config: TrainingPipelineConfig):
        # Data Validation Dir
        self.data_validation_dir = os.path.join(
            config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # Valid Data Dir and File Path
        self.valid_data_dir = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR_NAME
        )
        self.valid_train_file_path = os.path.join(
            self.valid_data_dir, training_pipeline.TRAIN_DATA_FILE_NAME
        )
        self.valid_test_file_path = os.path.join(
            self.valid_data_dir, training_pipeline.TEST_DATA_FILE_NAME
        )

        # Invalid Data Dir and File Path
        self.invalid_data_dir = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR_NAME
        )
        self.invalid_train_file_path = os.path.join(
            self.invalid_data_dir, training_pipeline.TRAIN_DATA_FILE_NAME
        )
        self.invalid_test_file_path = os.path.join(
            self.invalid_data_dir, training_pipeline.TEST_DATA_FILE_NAME
        )

        # Drift Report File Path
        self.drift_report_file_path = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    def __init__(self, config: TrainingPipelineConfig):
        # Data Transformation Dir
        self.data_transformation_dir = os.path.join(
            config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # Transformed Object Dir and FIle Path
        self.transformed_object_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME,
        )
        self.transformed_object_file_path = os.path.join(
            self.transformed_object_dir,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )

        # Transformed Data Dir and File Path
        self.transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME,
        )

        # Transformed Train and Test File Path
        self.transformed_train_file_path = os.path.join(
            self.transformed_data_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_NAME,
        )
        self.transformed_test_file_path = os.path.join(
            self.transformed_data_dir,
            training_pipeline.DATA_TRANSFORMATION_TEST_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, config: TrainingPipelineConfig):
        # Model Trainer Dir
        self.model_trainer_dir = os.path.join(
            config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )

        # Trained Model File Path
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR_NAME,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME,
        )

        # Model Trainer Report File Path
        self.training_report_file_path = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINING_REPORT_DIR_NAME,
            training_pipeline.MODEL_TRAINER_TRAINING_REPORT_FILE_NAME,
        )

        # Least Acceptable Accuracy
        self.expected_accuracy = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = (
            training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        )
