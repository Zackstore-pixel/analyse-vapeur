def compute_custom_correlation(df):
    import pandas as pd
    import numpy as np
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr(method='pearson')
    comments = ""

    if not corr.empty:
        strongest = corr.abs().unstack().sort_values(ascending=False)
        strongest = strongest[strongest < 1.0]  # Remove self-correlation
        top_corr = strongest.idxmax()
        comments = f"Highest correlation is between {top_corr[0]} and {top_corr[1]}: {corr.loc[top_corr[0], top_corr[1]]:.2f}"

    return corr.round(2), comments