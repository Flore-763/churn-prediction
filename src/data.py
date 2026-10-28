""" Chargement , nettoyage et feature engineering des données Telco Churn."""

from pathlib import Path
import pandas as pd 

DATA_PATH = Path("data/raw/telco_churn.csv")

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

def load_data(path: Path= DATA_PATH) -> pd.DataFrame:
    """Lit le CSV brut sans rien modifier."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base: types, colonnes inutiles, cible binaire."""
    df= df.copy()
    df=df.drop(columns=["customerID"]) # identifiant : aucune valeur prédictive

    # TotalCharges ets lu comme texte car 11 cases contiennent un espace
    # errors="coerce" transforme ces valeurs invalides en NaN

    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "Churn" in df.columns:
        df["Churn"]=df["Churn"].map({"Yes":1, "No":0})
    return df


def add_features(df:pd.DataFrame) ->  pd.DataFrame:
    """Crée de nouvelles varaibles à partir des existantes.
        Ces calculs se font ligne par ligne (aucune statistique globale).
        Il n'y a donc aucun risque de fuite de données (data leakage).
    """

    df = df.copy()

    #Nombre de services souscrits(0 à 8)

    df["n_services"]= (df[SERVICE_COLS]=="Yes").sum(axis=1)

    # Dépense mensuelle moyenne réelle (+1 pour eviter la division par 0)

    df["avg_monthly_spend"] = df["TotalCharges"]  /(df["tenure"]+1)

    # Ecart entre la facture actuelle et la moyenne historique

    df["charge_gap"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    return df
