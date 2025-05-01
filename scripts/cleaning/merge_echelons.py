def merge_echelon_sheets(file_path):
    import pandas as pd
    xl = pd.ExcelFile(file_path)
    sheets = xl.sheet_names
    dfs = [xl.parse(sheet).assign(Echelon=sheet) for sheet in sheets]
    return pd.concat(dfs, ignore_index=True)