import os
import numpy as np

# Common constants for the training pipeline
PIPELINE_NAME: str = "network_security"
TARGET_COLUMN: str = "Result"
ARTIFACTS_NAME: str = "artifacts"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")

FEATURE_STORE_FILE_NAME: str = "network_data.csv"
TRAIN_DATA_FILE_NAME: str = "train.csv"
TEST_DATA_FILE_NAME: str = "test.csv"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SAVED_MODEL_DIR_NAME = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"


# Data Ingestion related constants start with "DATA_INGESTION" variable name
DATA_INGESTION_DATABASE_NAME: str = "network_security"
DATA_INGESTION_COLLECTION_NAME: str = "network_data"
DATA_INGESTION_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str = "ingested"
DATA_INGESTION_TEST_SIZE: float = 0.2


# Data Validation related constants start with "DATA_VALIDATION" variable name
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR_NAME: str = "valid"
DATA_VALIDATION_INVALID_DIR_NAME: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"


# Data Transformation related constants start with "DATA_TRANSFORMATION" variable name
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME: str = "transformed_object"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test.npy"


# Model Trainer related constants start with "MODEL_TRAINER" variable name
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_TRAINING_REPORT_DIR_NAME: str = "training_report"
MODEL_TRAINER_TRAINING_REPORT_FILE_NAME: str = "model_trainer_report.yaml"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05
