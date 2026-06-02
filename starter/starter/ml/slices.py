import os
import pandas as pd
from ml.data import process_data
from ml.model import inference, compute_model_metrics

def compute_slices(df, feature_to_slice, cat_features, model, encoder, lb):
    """
    Computes performance metrics on slices of a given categorical feature
    and writes the final results directly to slice_output.txt.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../../slice_output.txt")
    
    with open(output_path, "w") as f:
        f.write(f"==================================================\n")
        f.write(f" PERFORMANCE SLICES FOR FEATURE: {feature_to_slice}\n")
        f.write(f"==================================================\n\n")
        
        # Ensure all cell data leading/trailing spaces are clean
        df_clean = df.copy()
        df_clean = df_clean.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        unique_vals = df_clean[feature_to_slice].unique()
        
        for val in unique_vals:
            # Isolate the data slice
            slice_df = df_clean[df_clean[feature_to_slice] == val]
            
            # Skip evaluation if the slice is empty
            if len(slice_df) == 0:
                continue
                
            # Process data using the pre-fitted encoder parameters
            X, y, _, _ = process_data(
                slice_df, 
                categorical_features=cat_features, 
                label="salary", 
                training=False, 
                encoder=encoder, 
                lb=lb
            )
            
            # Run model predictions and calculate specific metrics
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            
            # Log out formatting parameters
            f.write(f"Value Category: {val}\n")
            f.write(f"  Sample Size: {len(slice_df)}\n")
            f.write(f"  Precision:   {precision:.4f}\n")
            f.write(f"  Recall:      {recall:.4f}\n")
            f.write(f"  F1-Score:    {fbeta:.4f}\n")
            f.write("-" * 50 + "\n")
            
    print(f"Success! Slice performance metrics exported to: {os.path.abspath(output_path)}")
