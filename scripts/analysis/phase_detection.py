def detect_phases(df, columns_to_analyze, penalty=5, model="l2"):
    import ruptures as rpt
    import numpy as np

    for col in columns_to_analyze:
        series = df[col].copy()
        signal = series.fillna(method='ffill').fillna(method='bfill').values
        if len(signal) < 5:
            continue

        algo = rpt.Pelt(model=model).fit(signal)
        change_points = algo.predict(pen=penalty)

        phase_labels = []
        current_phase = 0
        idx = 0
        for cp in change_points:
            phase_labels.extend([current_phase] * (cp - idx))
            current_phase += 1
            idx = cp

        phase_col = f"Phase_{col.replace(' ', '_').replace('%', 'pct')}"
        df = df.iloc[:len(phase_labels)].copy()
        df[phase_col] = phase_labels

    return df
