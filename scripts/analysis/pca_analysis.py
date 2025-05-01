import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def run_pca(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    scaled_data = StandardScaler().fit_transform(df[numeric_cols])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    
    fig, ax = plt.subplots()
    ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6)
    ax.set_title("Projection PCA")
    return pca_df, fig
