def plot_single_variable(df, column):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['Timestamp'], df[column], label=column)
    ax.set_title(f"Time Series of {column}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(column)
    ax.grid(True)
    ax.legend()
    return fig