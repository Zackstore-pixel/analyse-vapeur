def generate_excel_report(df, filename="report.xlsx"):
    import pandas as pd
    import os
    with pd.ExcelWriter(os.path.join("output", filename)) as writer:
        df.describe().to_excel(writer, sheet_name="Descriptive Stats")
        if 'Cluster' in df.columns:
            df[['Cluster']].value_counts().to_frame("Count").to_excel(writer, sheet_name="Clusters")
        for col in df.columns:
            if col.startswith("Phase_"):
                df.groupby(col).mean().to_excel(writer, sheet_name=f"Stats_{col}")
    return filename
