import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(df: pd.DataFrame):
    """
    Runs PCA on numerical columns and returns the projection and a scatterplot figure.

    Returns:
        - pca_df: DataFrame with PCA projection (PC1, PC2)
        - fig: Matplotlib figure of scatterplot
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna()

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)

    # Combine with original index
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=numeric_df.index)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, c='blue')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Scatterplot")

    return pca_df, fig
