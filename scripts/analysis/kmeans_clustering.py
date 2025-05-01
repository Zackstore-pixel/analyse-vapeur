def run_kmeans(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)

    clustered_df = numeric_df.copy()
    clustered_df['Cluster'] = clusters

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    ax.set_title("KMeans Clustering (first 2 features)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    return clustered_df, fig