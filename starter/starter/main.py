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
                "fnlgt": 77516,
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
    # 1. Extract raw parameters using their exact hyphenated aliases
    input_dict = data.model_dump(by_alias=True)
    
    # 2. Create a clean DataFrame
    df_raw = pd.DataFrame([input_dict])
    
    ORIGINAL_COLUMNS_ORDER = [
        "age", "workclass", "fnlgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country"
    ]
    
    # Force the dataframe to take on the exact training sequence layout
    df = df_raw.reindex(columns=ORIGINAL_COLUMNS_ORDER)
    
    # Inject the dummy tracking target variable label
    df["salary"] = "<=5K"
    # 3. Explicitly enforce correct column types to prevent processing alignment issues
    int_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    # 4. Process features using your saved pipeline parameters safely
    try:
        X, _, _, _ = process_data(
            df, 
            categorical_features=CAT_FEATURES, 
            training=False, 
            encoder=encoder, 
            lb=lb
        )
        
        # 5. Execute model prediction
        raw_pred = inference(model, X)
        prediction_label = lb.inverse_transform(raw_pred)
        
        # 6. Extract the label value uniformly as a string clean of array brackets
        pred_str = str(prediction_label[0]) if hasattr(prediction_label, '__len__') and len(prediction_label) > 0 else str(prediction_label)
        return {"prediction": pred_str}
        
    except Exception as e:
        # This logs any hidden exception directly into your Render dashboard logs
        print(f"🔥 Prediction pipeline crashed internally: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")

