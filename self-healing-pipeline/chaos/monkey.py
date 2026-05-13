import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

def load_clean_data() -> pd.DataFrame:
    from utils.data import load_and_preprocess_data
    return load_and_preprocess_data()


def inject_nulls(df: pd.DataFrame, frac=0.15) -> pd.DataFrame:
    """Scenario A: inject nulls into key features."""
    corrupted = df.copy()
    for col in ["Amount", "V1", "V2"]:
        idx = corrupted.sample(frac=frac).index
        corrupted.loc[idx, col] = np.nan
    print(f"[Chaos] Injected nulls into Amount/V1/V2 ({frac*100:.0f}% rows)")
    return corrupted

def shift_distribution(df: pd.DataFrame, std_multiplier=2.5) -> pd.DataFrame:
    """Scenario B: shift Amount distribution to simulate concept drift."""
    corrupted = df.copy()
    shift = corrupted["Amount"].std() * std_multiplier
    corrupted["Amount"] = corrupted["Amount"] + shift
    print(f"[Chaos] Shifted Amount by +{shift:.2f} (concept drift)")
    return corrupted

def inject_outliers(df: pd.DataFrame, frac=0.05) -> pd.DataFrame:
    """Scenario C: inject extreme outliers into Amount."""
    corrupted = df.copy()
    idx = corrupted.sample(frac=frac).index
    corrupted.loc[idx, "Amount"] = 99999.0
    print(f"[Chaos] Injected outliers into Amount ({frac*100:.0f}% rows -> value=99999.0)")
    return corrupted

SCENARIOS = {
    "nulls":        inject_nulls,
    "drift":        shift_distribution,
    "outliers":     inject_outliers,
}

def run(scenario: str = "drift") -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(SCENARIOS)}")
    df = load_clean_data()
    corrupted = SCENARIOS[scenario](df)
    out_path = f"infra/corrupted_{scenario}.csv"
    corrupted.to_csv(out_path, index=False)
    print(f"[Chaos] Saved corrupted batch → {out_path}")
    return corrupted

if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "drift"
    run(scenario)