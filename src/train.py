"""Entraînement, recherche d'hyperparamètres, évaluation et sauvegarde."""


import json
from pathlib import Path
import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
 f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from src.data import add_features, clean_data, load_data
from src.preprocessing import build_preprocessor

RANDOM_STATE = 42
MODEL_PATH = Path("models/rf_churn.joblib")
METRICS_PATH = Path("models/metrics.json")
def main() -> None:
    # 1. Données
    df = add_features(clean_data(load_data()))
    X = df.drop(columns="Churn")
    y = df["Churn"]

    # 2. Split AVANT tout apprentissage (stratify garde le meme % de churn)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3. Pipeline complet : preprocessing + modèle
    pipe = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("model", RandomForestClassifier(
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    # 4. Recherche d'hyperparamètres avec validation croisée
    param_dist = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": randint(3, 20),
    "model__min_samples_leaf": randint(1, 30),
    "model__min_samples_split": randint(2, 30),
    "model__max_features": ["sqrt", "log2", 0.3, 0.5],
    }
    search = RandomizedSearchCV(
    pipe, param_dist, n_iter=30, scoring="roc_auc",
    cv=5, random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("Meilleurs paramètres :", search.best_params_)
    print(f"AUC (validation croisée) : {search.best_score_:.3f}")

    # 5. Évaluation finale sur le jeu de test (utilisé UNE seule fois)
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = {
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "recall_churn": round(recall_score(y_test, y_pred), 4),
    "f1_churn": round(f1_score(y_test, y_pred), 4),
    "cv_roc_auc": round(search.best_score_, 4),
    "best_params": {k.replace("model__", ""): v
    for k, v in search.best_params_.items()},
    }
    print(classification_report(y_test, y_pred, target_names=["Reste", "Part"]))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    # 6. Sauvegarde
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=int))
    print(f"Modèle sauvegardé dans {MODEL_PATH}")

    
if __name__ == "__main__":
 main()
