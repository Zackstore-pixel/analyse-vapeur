# ✅ scripts/analysis/influence_detection.py
import pandas as pd
import streamlit as st
# ✅ Détection de l'influence des variables sur la cible
def detect_variable_influence(df, targets):
    import pandas as pd

    result = {}
    df = df.select_dtypes(include=['float64', 'int64'])

    for target in targets:
        if target not in df.columns:
            continue
        correlations = df.corr()[target].drop(target).dropna()
        top_corr = correlations.abs().sort_values(ascending=False).head(5)
        result[target] = {
            "correlations": correlations.loc[top_corr.index],
            "top_variables": top_corr.index.tolist()
        }
    return result


def generate_interpretation(target, correlations):
    interpretations = []
    for var, value in correlations.items():
        trend = "augmente" if value > 0 else "diminue"
        interpretations.append(
            f"➡️ Quand **{var}** augmente, **{target}** {trend} (corrélation = {value:.2f})"
        )
    return interpretations


# ✅ Diagnostic de nettoyage (à insérer dans app.py après nettoyage)

def show_cleaning_diagnostic(df):
    import pandas as pd
    import numpy as np

    st.markdown("### 🧹 Diagnostic du nettoyage")
    st.markdown("""
    ✅ Données nettoyées automatiquement :
    - Suppression des valeurs manquantes (`NaN`, `no data`, etc.)
    - Normalisation des noms de colonnes
    - Tentative automatique de détection de la colonne temporelle
    """)

    st.markdown(f"- **Nombre de lignes** : {len(df)}")
    st.markdown(f"- **Nombre de colonnes** : {len(df.columns)}")
    nan_percent = df.isna().mean().sort_values(ascending=False) * 100
    with st.expander("🔍 Pourcentage de valeurs manquantes par variable"):
        st.dataframe(nan_percent.to_frame("% Manquants"))
