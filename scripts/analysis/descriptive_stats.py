def get_descriptive_stats(df, column):
    import pandas as pd
    stats = df[column].describe().to_frame(name='Value')
    stats.index.name = 'Metric'
    return stats.reset_index()