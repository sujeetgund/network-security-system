import os
import pandas as pd
import numpy as np
from pymongo.mongo_client import MongoClient
import certifi
from dotenv import load_dotenv
from uvicorn import run as app_run

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import RedirectResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.pipelines.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging import logger
from networksecurity.utils.main_utils import load_object


ca = certifi.where()
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI, tlsCAFile=ca)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
def train_route():
    try:
        training_pipeline_config = TrainingPipelineConfig()
        train_pipeline = TrainingPipeline(config=training_pipeline_config)
        train_pipeline.run_pipeline()
        return Response("Training was successful")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise NetworkSecurityException(e)


@app.post("/predict")
def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if df.empty:
            return Response("Empty dataset provided for prediction.")

        network_model: NetworkModel = load_object(
            filepath=os.path.join("final_model", "model.pkl")
        )

        preds = network_model.predict(df)
        if preds is None:
            return Response("Prediction failed. Please check the input data.")

        return JSONResponse(content={"predictions": preds.tolist()}, status_code=200)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise NetworkSecurityException(e)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
