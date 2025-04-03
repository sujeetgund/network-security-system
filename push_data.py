import os
import json
import pandas as pd
import numpy as np
import certifi
from dotenv import load_dotenv

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException


load_dotenv()

uri = os.getenv("MONGO_URI")

ca = certifi.where()


class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e)

    def csv_to_json(self, filepath: str):
        try:
            data = pd.read_csv(filepath)
            data.reset_index(drop=True, inplace=True)
            records = data.to_dict(orient="records")
            return records
        except Exception as e:
            raise NetworkSecurityException(e)

    def insert_data(
        self,
        records: list,
        db_name: str = "network_security",
        collection_name: str = "network_data",
    ):
        try:
            client = MongoClient(uri, server_api=ServerApi("1"))
            db = client[db_name]
            collection = db[collection_name]
            collection.insert_many(records)
            logger.info(f"Data inserted into {collection_name} collection.")
        except Exception as e:
            raise NetworkSecurityException(e)


if __name__ == "__main__":
    FILE_PATH = "network_data/phisingData.csv"

    data_extract = NetworkDataExtract()
    records = data_extract.csv_to_json(FILE_PATH)
    data_extract.insert_data(records)
