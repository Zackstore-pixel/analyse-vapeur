def plot_phases(df, column, phase_column):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 8))
    for phase in df[phase_column].unique():
        phase_data = df[df[phase_column] == phase]
        ax.plot(phase_data['Timestamp'], phase_data[column], label=f'Phase {phase}')
    ax.set_title(f'Phases - {column}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(column)
    ax.legend()
    ax.grid(True)
    return fig