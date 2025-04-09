from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.exception.exception import NetworkSecurityException


if __name__ == "__main__":
    try:
        # Initialize Training Pipeline Config
        training_pipeline_config = TrainingPipelineConfig()

        # Initialize Data Ingestion
        data_ingestion_config = DataIngestionConfig(config=training_pipeline_config)
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Get Data Ingestion Artifacts
        data_ingestion_artifact = data_ingestion.init_data_ingestion()

        # Initialize Data Validation
        data_validation_config = DataValidationConfig(config=training_pipeline_config)
        data_validation = DataValidation(
            config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact,
        )

        # Get Data Validation Artifacts
        data_validation_artifact = data_validation.init_data_validation()

        # Initialize Data Transformation
        data_transformation_config = DataTransformationConfig(
            config=training_pipeline_config
        )
        data_transformation = DataTransformation(
            config=data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        )

        # Get Data Transformation Artifacts
        data_transformation_artifact = data_transformation.init_data_transformation()
        print(data_transformation_artifact)
    except Exception as e:
        raise NetworkSecurityException(e)
