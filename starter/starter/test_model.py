import pytest
import numpy as np
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

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
    X_train = np.array([[39, 13, 1, 0, 0, 0, 0, 0, 0, 0], [50, 13, 0, 1, 0, 0, 0, 0, 0, 0], [38, 9, 0, 0, 1, 0, 0, 0, 0, 0]])
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
    X_test = np.array([[39, 13, 1, 0, 0, 0, 0, 0, 0, 0]])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)

def test_compute_model_metrics():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
