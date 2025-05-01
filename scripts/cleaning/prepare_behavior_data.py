def load_and_prepare_for_behavior_analysis(df):
    import pandas as pd
    import numpy as np

    # Nettoyage des colonnes : enlever les espaces, mettre en minuscules
    df.columns = df.columns.str.strip().str.lower()
    df.replace(["", "no data", "null", "NaN", "N/A"], np.nan, inplace=True)

    # Détection automatique d'une colonne temporelle
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() > len(df) * 0.6:
                df[col] = converted
                df.rename(columns={col: "timestamp"}, inplace=True)
                break
        except Exception:
            continue

    return df
