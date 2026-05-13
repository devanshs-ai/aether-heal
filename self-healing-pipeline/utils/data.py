import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data() -> pd.DataFrame:
    """Fetches the Credit Card Fraud dataset from OpenML and performs standard preprocessing."""
    
    # OpenML ID 1597 is the Credit Card Fraud Detection dataset
    print("Fetching Credit Card Fraud dataset from OpenML...")
    data = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    df = data.frame
    
    # The target is 'Class', make sure it's an integer
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)
        
    df.dropna(inplace=True)
    
    # Label Encoding for any potential categoricals (OpenML sometimes returns categorical types)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
        
    # Stratified sampling down to 50k rows for faster pipeline execution
    if len(df) > 50000:
        print(f"Sampling down from {len(df)} to 50,000 rows to ensure fast execution...")
        df, _ = train_test_split(df, train_size=50000, stratify=df["Class"], random_state=42)
        
    return df
