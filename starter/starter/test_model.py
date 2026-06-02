import sys
import os

# Find the absolute path to the repository root directory
current_dir = os.path.dirname(os.path.abspath(__file__)) # starter/starter
repo_root = os.path.abspath(os.path.join(current_dir, "../../")) # base directory

# Inject the repository root into the search path so starter.starter can be found
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from starter.starter import ml
sys.modules['ml'] = ml
sys.modules['ml.data'] = ml.data
sys.modules['ml.model'] = ml.model

import pytest
import numpy as np
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def sample_dataframe():
    data = {
        "age": [39, 50, 38],
        "workclass": [" State-gov", " Self-emp-not-inc", " Private"],
        "education": [" Bachelors", " Bachelors", " HS-grad"],
        "salary": [" <=5K", " >5K", " <=5K"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_trained_components():
    X_train = np.array([[39, 13], [50, 13], [38, 9]])
    y_train = np.array([0, 1, 0])
    model = train_model(X_train, y_train)
    return model

def test_process_data(sample_dataframe):
    cat_features = ["workclass", "education"]
    X, y, encoder, lb = process_data(sample_dataframe, categorical_features=cat_features, label="salary", training=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_train_model():
    X_dummy = np.array([[39, 13], [50, 13]])
    y_dummy = np.array([0, 1])
    model = train_model(X_dummy, y_dummy)
    assert model is not None
    assert hasattr(model, "predict")

def test_inference(mock_trained_components):
    model = mock_trained_components
    X_test = np.array([[39, 13], [50, 13]])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)

def test_compute_model_metrics():
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
