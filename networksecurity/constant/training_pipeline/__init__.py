import os

# Common constants for the training pipeline
PIPELINE_NAME: str = "network_security"
TARGET_COLUMN: str = "Result"
ARTIFACTS_NAME: str = "artifacts"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")

FEATURE_STORE_FILE_NAME: str = "network_data.csv"
TRAIN_DATA_FILE_NAME: str = "train.csv"
TEST_DATA_FILE_NAME: str = "test.csv"


# Data Ingestion related constants start with "DATA_INGESTION" variable name
DATA_INGESTION_DATABASE_NAME: str = "network_security"
DATA_INGESTION_COLLECTION_NAME: str = "network_data"
DATA_INGESTION_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_NAME: str = "ingested"
DATA_INGESTION_TEST_SIZE: float = 0.2


# Data Validation related constants start with "DATA_VALIDATION" variable name
DATA_VALIDATION_NAME: str = "data_validation"
DATA_VALIDATION_VALID_NAME: str = "valid"
DATA_VALIDATION_INVALID_NAME: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"
