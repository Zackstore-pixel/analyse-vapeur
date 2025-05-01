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

# ✅ Connexion avec session
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

# ✅ Interface principale après connexion
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

    if step == "1. Chargement & Nettoyage":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📄 Données brutes**")
            st.dataframe(raw_df.head(100), use_container_width=True)
        with col2:
            st.markdown("**🧼 Données nettoyées**")
            st.dataframe(df.head(100), use_container_width=True)
        show_cleaning_diagnostic(df)

    elif step == "2. Analyse Statistique Exploratoire (EDA)":
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if numeric_cols:
            selected_var = st.selectbox("Variable numérique à explorer :", numeric_cols)
            st.write(df[selected_var].describe().to_frame("Statistiques"))
            fig1, ax1 = plt.subplots()
            sns.histplot(df[selected_var], kde=True, ax=ax1)
            st.pyplot(fig1)
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[selected_var], ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("Aucune variable numérique trouvée.")

    elif step == "3. Corrélation":
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            st.dataframe(corr_matrix)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            redundant = corr_matrix[(corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)].stack().reset_index()
            redundant.columns = ["Variable 1", "Variable 2", "Corrélation"]
            if not redundant.empty:
                st.dataframe(redundant)
                st.info("💡 PCA recommandée pour réduire la redondance.")
        else:
            st.warning("Pas assez de variables numériques pour calculer les corrélations.")

    elif step == "4. Réduction de Dimension (PCA)":
        pca_df, fig = run_pca(df)
        st.pyplot(fig)
        st.dataframe(pca_df.head())

    elif step == "5. Clustering K-Means":
        cluster_df, fig = run_kmeans(df)
        st.pyplot(fig)
        st.dataframe(cluster_df[["Cluster"]].value_counts().to_frame("Nb d'observations"))

    elif step == "6. Interprétation & Recommandations":
        st.subheader("🧾 Résumé & Interprétations")
        st.markdown("**📌 Ce que nous avons appris :**")
        st.markdown("- Nettoyage, analyse, corrélation, PCA, clustering")

        st.markdown("**🔍 Recommandations possibles :**")
        st.markdown("- Fort débit vapeur + basse température → gaspillage thermique")
        st.markdown("- Faible débit + température stable → fonctionnement optimal")
        st.markdown("- Suivi en temps réel des variables les plus corrélées au débit")

        if 'timestamp' in df.columns:
            time_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            selected_time_var = st.selectbox("📈 Variable à visualiser sur le temps :", time_cols)

            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=["timestamp"])

            min_date, max_date = df['timestamp'].min(), df['timestamp'].max()
            date_range = st.slider("Plage temporelle :", min_value=min_date, max_value=max_date,
                                   value=(min_date, max_date), format="YYYY-MM-DD HH:mm")

            filtered_df = df[(df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])]

            fig_time, ax_time = plt.subplots(figsize=(12, 4))
            ax_time.plot(filtered_df['timestamp'], filtered_df[selected_time_var], color='tab:blue')
            ax_time.set_title(f"{selected_time_var} en fonction du temps")
            ax_time.set_xlabel("Temps")
            ax_time.set_ylabel(selected_time_var)
            st.pyplot(fig_time)
        else:
            st.warning("Colonne temporelle non détectée.")

else:
    st.info("💡 Merci d'importer un fichier Excel pour démarrer l’analyse.")
