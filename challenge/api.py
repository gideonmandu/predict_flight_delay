import os
from asyncio.log import logger

import fastapi
from fastapi import HTTPException, status
from challenge.model import DelayModel
import pandas as pd
from pydantic import BaseModel
from typing import List

app = fastapi.FastAPI()

# Initialize the DelayModel
model_instance = DelayModel()
# load model
model_instance.load_model(filepath="models/model.joblib")
# check if model is loaded
if model_instance._model is None:
    # load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'data.csv')
    data = pd.read_csv(
        filepath_or_buffer=data_path.replace('challenge/', '', 1),
        low_memory=False
    )
    # preprocess data
    features, target = model_instance.preprocess(
        data=data, target_column="delay"
    )
    # fit model
    model_instance.fit(features=features, target=target)
    # save model
    model_instance.save_model(filepath="models/model.joblib")
    # load model
    model_instance.load_model(filepath="models/model.joblib")


class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class Payload(BaseModel):
    flights: List[FlightData]


class Prediction(BaseModel):
    predict: List[int]


class ErrorResponse(BaseModel):
    detail: str


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post(
    "/predict",
    status_code=status.HTTP_200_OK,
    response_model=Prediction,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    },
)
async def post_predict(payload: Payload) -> dict:
    """
    Predict delays using the DelayModel.
    """
    # Convert incoming data to DataFrame format
    df = pd.DataFrame([flight.model_dump() for flight in payload.flights])

    # Validation checks for all flights
    if not df["MES"].between(1, 12).all():
        logger.error("Invalid MES value detected")
        raise HTTPException(status_code=400, detail="Invalid MES value.")
    if not df["TIPOVUELO"].isin(["N", "I"]).all():
        logger.error("Invalid TIPOVUELO value detected")
        raise HTTPException(status_code=400, detail="Invalid TIPOVUELO value.")
    try:
        # Preprocess the data
        data_features = model_instance.preprocess(data=df)
        # Predict the delay
        predictions = model_instance.predict(features=data_features)
    except ValueError as e:
        logger.error(f"Error during preprocessing or prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"predict": predictions}
