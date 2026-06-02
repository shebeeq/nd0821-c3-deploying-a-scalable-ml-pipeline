
import sys
import os
from fastapi.testclient import TestClient
# Forces the CI runner to look in the exact directory where your code lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../..")))

# Ensure the local directory path can be tracked by pytest natively
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app

client = TestClient(app)

# --- TEST 1: GET Method Verification ---
def test_get_root():
    """
    Tests GET on the root domain.
    Verifies both the 200 HTTP status code and the exact dictionary greeting.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome to the Census Income Predictive Model Portal!"}


# --- TEST 2: POST Inference for Low Income Bracket (<=5K) ---
def test_post_predict_low_income():
    """
    Tests POST inference targeting an expected '<=5K' income bracket output classification.
    """
    payload = {
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
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Checks that a prediction payload key exists and asserts low-income bracket matching
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "['<=5K']"


# --- TEST 3: POST Inference for High Income Bracket (>5K) ---
def test_post_predict_high_income():
    """
    Tests POST inference targeting an expected '>5K' income bracket output classification.
    """
    payload = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "fnlwgt": 83311,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 99999,  # High capital gain helps trigger high-income classification
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Checks that a prediction payload key exists and asserts high-income bracket matching
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "['>5K']"
