import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def run_kmeans(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    scaled_data = StandardScaler().fit_transform(df[numeric_cols])
    pca_data = PCA(n_components=2).fit_transform(scaled_data)
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(pca_data)

    df["Cluster"] = labels
    fig, ax = plt.subplots()
    ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap="Set1")
    ax.set_title("Clusters K-Means (PCA)")
    return df, fig
