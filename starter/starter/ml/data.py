import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """
    Cleans trailing spaces, handles OneHotEncoding for categorical features,
    and binarizes target labels safely for both training and live inference.
    """
    X = X.copy()

    # Clean the messy data: remove all leading/trailing whitespaces from text columns
    X = X.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    if label is not None and label in X.columns:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # Separate continuous and categorical features safely
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features, errors='ignore').values

    # Ensure continuous data is a valid NumPy array and not None
    if X_continuous is None or (hasattr(X_continuous, "ndim") and X_continuous.ndim == 0):
        X_continuous = np.empty((X.shape[0], 0))

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        if label is not None and len(y) > 0:
            y = lb.fit_transform(y).ravel()
    else:
        # Live Inference Pipeline Mode
        if encoder is not None:
            X_categorical = encoder.transform(X_categorical)
        else:
            X_categorical = np.empty((X.shape[0], 0))
            
        if label is not None and len(y) > 0:
            try:
                if lb is not None and hasattr(lb, "classes_") and lb.classes_ is not None:
                    y = lb.transform(y).ravel()
                else:
                    y = np.array([])
            except Exception:
                y = np.array([])
        else:
            y = np.array([])

    # Force continuous and categorical matrices to be true NumPy arrays to prevent 'NoneType' crashes
    X_continuous = np.asarray(X_continuous)
    X_categorical = np.asarray(X_categorical)

    # Perform the final alignment concatenation safely
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    
    return X, y, encoder, lb
