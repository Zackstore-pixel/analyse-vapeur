# ✅ Analyseur Intelligent - Données Vapeur (Assistant Étape par Étape)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
def check_login():
    st.sidebar.markdown("## 🔐 Connexion requise")
    username = st.sidebar.text_input("Utilisateur", value="", placeholder="Entrez votre nom")
    password = st.sidebar.text_input("Mot de passe", value="", type="password")

    if username == "encadrant" and password == "ocp2025":
        return True
    else:
        st.warning("Identifiants requis ou incorrects.")
        return False

if not check_login():
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
    df = load_and_prepare_for_behavior_analysis(raw_df.copy())
    df = df.fillna(method="ffill").fillna(method="bfill")

    # 🔍 Filtrage par échelon
    if 'echelon' in df.columns:
        unique_echelons = df['echelon'].dropna().unique().tolist()
        selected_echelon = st.sidebar.selectbox("🔍 Choisir un échelon à analyser :", sorted(unique_echelons))
        df = df[df['echelon'] == selected_echelon]
        st.success(f"✅ Analyse ciblée sur l’échelon : {selected_echelon}")
    else:
        st.warning("⚠️ Colonne 'echelon' non trouvée — l’analyse se fera sur toutes les données.")

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
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) >= 2:
            scaled = StandardScaler().fit_transform(df[numeric_cols])
            pca = PCA().fit(scaled)
            var_explained = pca.explained_variance_ratio_.cumsum()
            fig, ax = plt.subplots()
            ax.plot(range(1, len(var_explained)+1), var_explained, marker="o")
            ax.set_title("Variance Cumulative Expliquée")
            st.pyplot(fig)

            n_components = st.slider("Nombre de composantes :", 2, min(10, len(numeric_cols)), 2)
            pca_reduced = PCA(n_components=n_components).fit_transform(scaled)
            pca_df = pd.DataFrame(pca_reduced, columns=[f"PC{i+1}" for i in range(n_components)])
            st.dataframe(pca_df.head())

            if n_components >= 2:
                fig2d, ax2d = plt.subplots()
                ax2d.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6)
                ax2d.set_title("Projection 2D des données PCA")
                st.pyplot(fig2d)
        else:
            st.warning("PCA non applicable : pas assez de variables.")

    elif step == "5. Clustering K-Means":
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(numeric_cols) >= 2:
            scaled = StandardScaler().fit_transform(df[numeric_cols])
            reduced = PCA(n_components=2).fit_transform(scaled)
            distortions = [KMeans(n_clusters=k, random_state=42).fit(reduced).inertia_ for k in range(1, 11)]
            fig, ax = plt.subplots()
            ax.plot(range(1, 11), distortions, marker='o')
            ax.set_title("Méthode du coude (Elbow method)")
            st.pyplot(fig)

            k = st.slider("Choisir le nombre de clusters :", 2, 10, 3)
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(reduced)

            fig2, ax2 = plt.subplots()
            ax2.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="Set1")
            ax2.set_title("Clusters K-Means sur projection PCA")
            st.pyplot(fig2)

            st.bar_chart(pd.Series(labels).value_counts().sort_index())
        else:
            st.warning("Clustering non applicable.")

    elif step == "6. Interprétation & Recommandations":
        st.subheader("🧾 Résumé & Interprétations")
        st.markdown("**📌 Ce que nous avons appris :**")
        st.markdown("- Nettoyage, analyse, corrélation, PCA, clustering")

        st.markdown("**🔍 Recommandations possibles :**")
        st.markdown("- Fort débit vapeur + basse température → gaspillage thermique")
        st.markdown("- Faible débit + température stable → fonctionnement optimal")
        st.markdown("- Suivi en temps réel des variables les plus corrélées au débit")
#
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])  # Supprimer les lignes sans timestamp
                time_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
                selected_time_var = st.selectbox("📈 Variable à visualiser sur le temps :", time_cols)
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                if pd.isnull(min_date) or pd.isnull(max_date):
                    st.warning("⛔ Dates invalides détectées dans la colonne 'timestamp'.")
                else:
                    min_date = pd.to_datetime(min_date).to_pydatetime()
                    max_date = pd.to_datetime(max_date).to_pydatetime()
                    
                    date_range = st.slider("Plage temporelle :", min_value=min_date, max_value=max_date,
                                   value=(min_date, max_date), format="YYYY-MM-DD HH:mm")
                    
                    filtered_df = df[(df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])]

                    fig_time, ax_time = plt.subplots(figsize=(12, 4))
                    ax_time.plot(filtered_df['timestamp'], filtered_df[selected_time_var], color='tab:blue')
                    ax_time.set_title(f"{selected_time_var} en fonction du temps")
                    ax_time.set_xlabel("Temps")
                    ax_time.set_ylabel(selected_time_var)
                    ax_time.grid(True)
                    st.pyplot(fig_time)

            except Exception as e:
                st.error(f"Erreur lors du traitement de la courbe temporelle : {e}")
        else:
            st.warning("Colonne temporelle non détectée.")

#
        st.markdown("**📥 Rapport automatique en cours de développement.**")

else:
    st.info("💡 Merci d'importer un fichier Excel pour démarrer l’analyse.")
