import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

DATA_URL = os.getenv(
    "DATA_URL", 
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

def load_and_preprocess_data() -> pd.DataFrame:
    """Fetches the dataset and performs standard preprocessing."""
    df = pd.read_csv(DATA_URL)
    
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
    df.dropna(inplace=True)
    
    # Label Encoding for categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
        
    return df
