from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    """ Tests GET on the root domain greeting contents. """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome to the Census Income Predictive Model Portal!"}

def test_post_predict_low_income():
    """ Tests POST inference targeting an expected '<=5K' bracket. """
    payload = {
        "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors",
        "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "<=5K"

def test_post_predict_high_income():
    """ Tests POST inference targeting an expected '>5K' bracket. """
    payload = {
        "age": 50, "workclass": "Self-emp-not-inc", "fnlwgt": 83311, "education": "Bachelors",
        "education-num": 13, "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital-gain": 99999, "capital-loss": 0, "hours-per-week": 50, "native-country": "United-States"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == ">5K"
