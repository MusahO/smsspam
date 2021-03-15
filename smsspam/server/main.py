from typing import List
import logging
import os

import numpy as np
from fastapi import FastAPI
from pydantic.main import BaseModel
from pydantic import BaseSettings
from fastapi.middleware.cors import CORSMiddleware

from utils.features import Datapoint

from model.boosting import LGBMModel

from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()  # load environment variables


logging.basicConfig(format='%(levelname)s--%(asctime)s--%(filename)s--%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
twilio_api = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    model_dir : str = os.path.join(base_dir, 'model/checkpoints/lgbm')

app = FastAPI()
settings = Settings()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = {
    "model_output_path": settings.model_dir,
    "featurizer_output_path": settings.model_dir,
    "params" : {}
}

model = LGBMModel(config)

class SMS(BaseModel):
    text: str

class Predictions(BaseModel):
    label: float
    proba: List[float]

@app.post("/api/spam-sms", response_model=Predictions)
def predict(sms: SMS):
    datapoint = Datapoint(**{"sms": sms.text, "sms_length": int(float("nan"))})
    proba = model.predict([datapoint])
    label = np.argmax(proba, axis=1)
    prediction = Predictions(label=label[0], proba=list(proba[0]))
    logger.info(prediction)
    return prediction