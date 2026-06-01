# Put the code for your API here.
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI(
    title="Census Income Inference API",
    description="An API to predict structural income brackets from census characteristics."
)

# Setup pathing variables relative to this module root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../")) # Base repo directory

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

# Point directly to the model folder in the base repository root directory
MODEL_DIR = os.path.join(REPO_ROOT, "model")
print(f"Loading tracking models directly from: {os.path.abspath(MODEL_DIR)}")
# 1. Try checking inside starter/starter/model/

# Load our saved pipeline artifacts globally on app startup
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)
with open(os.path.join(MODEL_DIR, "lb.pkl"), "rb") as f:
    lb = pickle.load(f)

# Track our established structural features
CAT_FEATURES = [
    "workclass", "education", "marital-status", 
    "occupation", "relationship", "race", "sex", "native-country"
]

class CensusData(BaseModel):
    """ Data transfer object utilizing field aliasing to digest hyphens cleanly. """
    age: int = Field(..., description="Age of the individual")
    workclass: str = Field(..., description="Employment classification status")
    fnlwgt: int = Field(..., description="Final weight variable representation")
    education: str = Field(..., description="Highest completed educational milestone")
    education_num: int = Field(..., alias="education-num", description="Numeric code for education status")
    marital_status: str = Field(..., alias="marital-status", description="Marital status classification")
    occupation: str = Field(..., description="Work industry role profile description")
    relationship: str = Field(..., description="Household layout demographic metric")
    race: str = Field(..., description="Individual race demographic profile description")
    sex: str = Field(..., description="Biological sex property assignment details")
    capital_gain: int = Field(..., alias="capital-gain", description="Monetary tracking gains record values")
    capital_loss: int = Field(..., alias="capital-loss", description="Monetary tracking loss metrics data")
    hours_per_week: int = Field(..., alias="hours-per-week", description="Work commitment metrics data representation")
    native_country: str = Field(..., alias="native-country", description="Geographic origin profiling details")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
    }

@app.get("/")
async def get_root():
    """ GET request on the root domain returning a welcome greeting. """
    return {"greeting": "Welcome to the Census Income Predictive Model Portal!"}

@app.post("/predict")
async def post_predict(data: CensusData):
    """ POST request running end-to-end model inference on an incoming feature vector. """
    # Ingest using our raw hyphenated parameters via dump alias extraction
    input_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])
    
    # Process features using the saved operational pipeline parameters
    X, _, _, _ = process_data(
        df, 
        categorical_features=CAT_FEATURES, 
        training=False, 
        encoder=encoder, 
        lb=lb
    )
    
    # Run structural prediction checks
    raw_pred = inference(model, X)
    prediction_label = lb.inverse_transform(raw_pred)[0]
    
    return {"prediction": str(prediction_label)}
