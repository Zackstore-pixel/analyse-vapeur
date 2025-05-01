import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ✅ Optimisations Render gratuites
os.environ["BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scripts.cleaning.prepare_behavior_data import load_and_prepare_for_behavior_analysis
from scripts.analysis.descriptive_stats import get_descriptive_stats
from scripts.analysis.correlation_analysis import compute_custom_correlation
from scripts.analysis.pca_analysis import run_pca
from scripts.analysis.kmeans_clustering import run_kmeans
from scripts.analysis.phase_detection import detect_phases
from scripts.analysis.influence_detection import detect_variable_influence, generate_interpretation, show_cleaning_diagnostic
from scripts.visualization.plot_variable import plot_single_variable
from scripts.visualization.plot_phases import plot_phases
from scripts.utils.io_helpers import save_to_excel

st.set_page_config(page_title="Assistant Analyse Vapeur", layout="wide")

#✅ Connexion avec session
def login_form():
    with st.sidebar.form(key="login_form"):
        st.markdown("## 🔐 Connexion requise")
        username = st.text_input("Identifiant", placeholder="ex: encadrant")
        password = st.text_input("Mot de passe", type="password", placeholder="••••••")
        submitted = st.form_submit_button("Se connecter")
        if submitted:
            if username == "encadrant" and password == "ocp2025":
                st.session_state.logged_in = True
            else:
                st.error("Identifiants incorrects.")

# Vérifie la session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_form()
    st.stop()

st.title("🧠 Assistant Intelligent - Analyse de Données Vapeur")

step = st.sidebar.radio("Étapes de l’analyse", [
    "1. Chargement & Nettoyage",
    "2. Analyse Statistique Exploratoire (EDA)",
    "3. Corrélation",
    "4. Réduction de Dimension (PCA)",
    "5. Clustering K-Means",
    "6. Interprétation & Recommandations"
])

uploaded_file = st.sidebar.file_uploader("📁 Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)

    with st.spinner("Chargement intelligent des données..."):
        df = load_and_prepare_for_behavior_analysis(raw_df.copy())
        df = df.fillna(method="ffill").fillna(method="bfill")

    # 🧠 Reste de ton code logique ici (EDA, PCA, etc.)

else:
    st.info("💡 Merci d'importer un fichier Excel pour démarrer l’analyse.")
