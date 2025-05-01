def save_to_excel(df, filename="output.xlsx", folder="output"):
    import os
    path = os.path.join(folder, filename)
    df.to_excel(path, index=False)
    return path
