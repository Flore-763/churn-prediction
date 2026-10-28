"""Pipeline de preprocessing réutilisable """
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le ColumnTransformer à partir des types de colonnes de X."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_pipe = Pipeline([
    # Pas de scaling : les arbres ne dépendent pas de l'échelle des variables
    ("imputer", SimpleImputer(strategy="median")),
    ])  # les valeurs manquantes sont remplacées par la mediane(car elle n'est pas influencée par les valers aberrantes ce qui est le cas pour la moyenne)
    
    cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    return ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
    ])