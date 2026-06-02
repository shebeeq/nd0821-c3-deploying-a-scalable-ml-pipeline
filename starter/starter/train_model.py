from ml.slices import compute_slices
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local absolute file path alignment imports

# 1. Load the raw census dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/census.csv")
data = pd.read_csv(data_path)

# Strip spaces from the column names to prevent KeyErrors
data.columns = data.columns.str.strip()

# Print the clean column names to your console for visual verification
print("Cleaned Columns found:", list(data.columns))

# 2. Define categorical properties
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# 3. Implement a clean Train-Test Split (80% Train, 20% Test)
train, test = train_test_split(data, test_size=0.20, random_state=42)

# 4. Preprocess the training subset
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# 5. Preprocess the testing subset using the fitted encoder models
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# 6. Train the machine learning model
print("Training model...")
model = train_model(X_train, y_train)

# 7. Evaluate model performance metrics
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(
    f"Test Metrics -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {fbeta:.4f}")

# 8. Save the model, encoder, and label binarizer artifacts to disk
model_dir = os.path.join(os.path.dirname(__file__), "../../model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(model_dir, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)
with open(os.path.join(model_dir, "lb.pkl"), "wb") as f:
    pickle.dump(lb, f)

# 9. Compute and output feature slice metrics (e.g., on the 'education' feature)
print("Computing model slices for the 'education' feature...")
compute_slices(test, "education", cat_features, model, encoder, lb)

print("All pipeline artifacts successfully exported to /model/ directory!")
