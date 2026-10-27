"""Chargement, nettoyage et feature engineering des données Telco Churn."""
from pathlib import Path
import pandas as pd
DATA_PATH = Path("../data/raw/telco_churn.csv")
SERVICE_COLS = [
 "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
 "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lit le CSV brut sans rien modifier."""
    return pd.read_csv(path)
