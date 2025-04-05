from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException


if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()

        # Initialize Data Ingestion Config
        data_ingestion_config = DataIngestionConfig(config=training_pipeline_config)
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Get Data Ingestion Artifacts
        data_ingestion_artifacts = data_ingestion.init_data_ingestion()
        print(data_ingestion_artifacts)
    except Exception as e:
        raise NetworkSecurityException(e)
