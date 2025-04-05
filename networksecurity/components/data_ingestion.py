from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import numpy as np
import pandas as pd
import certifi
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.model_selection import train_test_split


load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

ca = certifi.where()


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
        except Exception as e:
            raise NetworkSecurityException(e)

    def export_collection_into_df(self) -> pd.DataFrame:
        try:
            database_name = self.config.database_name
            collection_name = self.config.collection_name

            logger.info("Connecting to MongoDB")
            mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
            db = mongo_client[database_name]
            logger.info(f"Connected to database: {database_name}")
            collection = db[collection_name]
            logger.info(f"Connected to collection: {collection_name}")

            data = list(collection.find())
            df = pd.DataFrame(data)

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)

            return df
        except Exception as e:
            raise NetworkSecurityException(e)

    def export_df_into_feature_store(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            df.to_csv(feature_store_file_path, index=False)
            logger.info(f"Data exported to feature store at {feature_store_file_path}")
            return df
        except Exception as e:
            raise NetworkSecurityException(e)

    def split_data_into_train_test(self, df: pd.DataFrame) -> None:
        try:
            logger.info("Splitting data into train and test sets")
            logger.info(f"Test size: {self.config.test_size}")
            train_data, test_data = train_test_split(
                df, test_size=self.config.test_size, random_state=42
            )

            train_data_path = self.config.training_file_path
            test_data_path = self.config.testing_file_path

            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

            logger.info("Exporting train and test data")
            train_data.to_csv(train_data_path, index=False)
            test_data.to_csv(test_data_path, index=False)
        except Exception as e:
            raise NetworkSecurityException(e)

    def init_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting data ingestion process")
        try:
            df = self.export_collection_into_df()
            feature_store_df = self.export_df_into_feature_store(df)
            self.split_data_into_train_test(feature_store_df)

            artifact = DataIngestionArtifact(
                train_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path,
            )
            logger.info("Data ingestion completed successfully")

            return artifact

        except Exception as e:
            raise NetworkSecurityException(e)
