import json
import requests

# 1. Swap this string out with your real public production web address
LIVE_URL = "https://nd0821-c3-deploying-a-scalable-ml.onrender.com/predict"

# 2. Setup standard test feature payload properties matching Pydantic expectation
payload = {
    "age": 45,
    "workclass": "Private",
    "fnlwgt": 150000,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

print(f"Sending live cloud POST query connection request to: {LIVE_URL}\n")

# 3. Deliver requests tracking data
response = requests.post(LIVE_URL, json=payload)

# 4. Return results inside your local execution terminal
print(f"HTTP Return Status Code: {response.status_code}")
print(f"Raw Response Text: {response.text}") 
# print(f"API Model Inference Response Result: {response.json()}")
