"""Application Streamlit : prédiction du churn client."""
import json
import joblib
import pandas as pd
import streamlit as st
from src.data import add_features, clean_data, load_data


st.set_page_config(page_title="Churn Predictor", layout="wide")

# ---------- Chargement (mis en cache : exécuté une seule fois) ----------
@st.cache_resource
def load_model():
    return joblib.load("models/rf_churn.joblib")

@st.cache_data
def load_dataset() -> pd.DataFrame:
    return load_data()

@st.cache_data
def load_metrics() -> dict:
 with open("models/metrics.json") as f:
    return json.load(f)
 
model = load_model()

st.title("Prédiction du départ client (churn)")
tab_pred, tab_data, tab_model = st.tabs(
 ["Prédire", "Explorer les données", "À propos du modèle"])

# ---------- Barre latérale : profil du client ----------
with st.sidebar:
    st.header("Profil du client")
    tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    monthly = st.slider("Facture mensuelle ($)", 18.0, 120.0, 70.0, 0.5)
    contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Paiement", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.radio("Facture dématérialisée", ["Yes", "No"], horizontal=True)
    senior = st.checkbox("Senior (65 ans et plus)")
    partner = st.radio("En couple", ["Yes", "No"], horizontal=True)
    dependents = st.radio("Personnes à charge", ["Yes", "No"], horizontal=True)
    phone = st.radio("Service téléphone", ["Yes", "No"], horizontal=True)
    # Les options dépendent de la souscription internet / téléphone
    opt = ["Yes", "No"] if internet != "No" else ["No internet service"]


with st.expander("Services optionnels"):
    lines = st.selectbox(
    "Lignes multiples",
    ["Yes", "No"] if phone == "Yes" else ["No phone service"])
    security = st.selectbox("Sécurité en ligne", opt)
    backup = st.selectbox("Sauvegarde en ligne", opt)
    protection = st.selectbox("Protection appareil", opt)
    support = st.selectbox("Support technique", opt)
    tv = st.selectbox("Streaming TV", opt)
    movies = st.selectbox("Streaming films", opt)


def build_input() -> pd.DataFrame:
    """Construit UNE ligne avec exactement les colonnes brutes du CSV."""
    row = {
    "customerID": "demo", "gender": "Female", "SeniorCitizen": int(senior),
    "Partner": partner, "Dependents": dependents, "tenure": tenure,
    "PhoneService": phone, "MultipleLines": lines,
    "InternetService": internet, "OnlineSecurity": security,
    "OnlineBackup": backup, "DeviceProtection": protection,
    "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
    "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": str(round(monthly * tenure, 2)) if tenure > 0 else " ",
    }
    # Mêmes transformations que pendant l'entraînement
    return add_features(clean_data(pd.DataFrame([row])))


# ---------- Onglet 1 : prédiction ----------
with tab_pred:
    X_new = build_input()
    proba = float(model.predict_proba(X_new)[0, 1])
    col1, col2 = st.columns([1, 2])
    col1.metric("Probabilité de départ", f"{proba:.0%}")
    col2.progress(proba)
    if proba >= 0.6:
        st.error("Risque élevé : action de fidélisation recommandée.")
    elif proba >= 0.35:
        st.warning("Risque modéré : à surveiller.")
    else:
        st.success("Risque faible.")
    with st.expander("Voir les données envoyées au modèle"):
        st.dataframe(X_new)


# ---------- Onglet 2 : exploration ----------
with tab_data:
    df = load_dataset()
    col = st.selectbox("Variable à analyser", [
    "Contract", "InternetService", "PaymentMethod", "SeniorCitizen"])
    rate = (df.assign(churn=df["Churn"].eq("Yes"))
    .groupby(col)["churn"].mean().sort_values())
    st.subheader(f"Taux de churn par {col}")
    st.bar_chart(rate)
    st.dataframe(df.head(50))

# ---------- Onglet 3 : modèle ----------
with tab_model:
    m = load_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC AUC (test)", m["roc_auc"])
    c2.metric("Recall (churn)", m["recall_churn"])
    c3.metric("F1 (churn)", m["f1_churn"])
    st.subheader("Hyperparamètres retenus")
    st.json(m["best_params"])
    names = model["prep"].get_feature_names_out()
    imp = pd.Series(model["model"].feature_importances_, index=names)
    st.subheader("Variables les plus importantes")
    st.bar_chart(imp.nlargest(10).sort_values())