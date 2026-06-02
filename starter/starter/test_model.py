import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Import your core module functions
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def sample_dataframe():
    """ Fixture providing a mock DataFrame for testing data processing. """
    data = {
        "age": [39, 50, 38],
        "workclass": [" State-gov", " Self-emp-not-inc", " Private"],  # Messy spaces included
        "education": [" Bachelors", " Bachelors", " HS-grad"],
        "salary": [" <=5K", " >5K", " <=5K"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_trained_components():
    """ Fixture mimicking a simple pre-trained tracking setup. """
    X_train = np.array([[39, 1, 0], [50, 0, 1], [38, 0, 0]])
    y_train = np.array([0, 1, 0])
    model = train_model(X_train, y_train)
    return model


# --- TEST 1: process_data function ---
def test_process_data(sample_dataframe):
    """ Test that data processing correctly cleans strings and outputs the expected types. """
    cat_features = ["workclass", "education"]
    
    X, y, encoder, lb = process_data(
        sample_dataframe, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    
    # Check that output objects match expected types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
    
    # Assert row dimension consistency matches input DataFrame size
    assert X.shape[0] == len(sample_dataframe)
    assert len(y) == len(sample_dataframe)


# --- TEST 2: train_model function ---
def test_train_model():
    """ Test that model training initializes and outputs a fitted Scikit-Learn classifier. """
    X_dummy = np.array([[25, 1], [45, 0], [35, 1], [55, 0]])
    y_dummy = np.array([0, 1, 0, 1])
    
    model = train_model(X_dummy, y_dummy)
    
    assert model is not None
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")  # Confirms it is an estimator package


# --- TEST 3: inference function ---
def test_inference(mock_trained_components):
    """ Test that inference delivers predictions matching the sample dimension constraints. """
    model = mock_trained_components
    X_test = np.array([[40, 1, 0], [32, 0, 0]])
    
    preds = inference(model, X_test)
    
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)
    # Binary labels must be either 0 or 1
    assert all(p in [0, 1] for p in preds)


# --- TEST 4: compute_model_metrics function ---
def test_compute_model_metrics():
    """ Test that the metric evaluator returns float values bounded between 0 and 1. """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
