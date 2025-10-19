
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

if len(sys.argv) < 3:
    print("Usage: python strong_weakTimeComp.py [strong|weak] <csv_file>")
    sys.exit(1)

scaling_type = sys.argv[1]
csv_file = sys.argv[2]

# Leggi il CSV
df = pd.read_csv(csv_file)

# Filtra le 4 fasi di interesse

phases = ['Total time (iterations)', 'Update']
df = df[df['Phase'].isin(['Total Iterations', 'MPI Setup', 'Wait', 'Update'])]


# Ordina i nodi come richiesto (senza 6)
node_order = [1, 2, 4, 8, 16]
df = df[df['NODES'].isin(node_order)]
df['NODES'] = df['NODES'].astype(int)
df = df.sort_values('NODES')


# Prepara i dati per il line plot
phase_colors = {
    'Total time (iterations)': '#1f77b4',
    'Wait+MPI Setup': '#ff7f0e',
    'Update': '#d62728'
}

fig, ax = plt.subplots(figsize=(8, 6))


for phase in phases:
    # Per la curva 'Total time (iterations)' usa i dati di 'Total Iterations'
    if phase == 'Total time (iterations)':
        phase_data = df[df['Phase'] == 'Total Iterations']
    else:
        phase_data = df[df['Phase'] == phase]
    means = []
    mins = []
    maxs = []
    for n in node_order:
        row = phase_data[phase_data['NODES'] == n]
        if not row.empty:
            means.append(row['Avg'].values[0])
            mins.append(row['Min'].values[0])
            maxs.append(row['Max'].values[0])
        else:
            means.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    means = np.array(means, dtype=float)
    mins = np.array(mins, dtype=float)
    maxs = np.array(maxs, dtype=float)
    ax.plot(node_order, means, marker='o', label=phase, color=phase_colors[phase], zorder=2)

# Calcola la curva Wait+MPI Setup
means_sum = []
for n in node_order:
    wait_row = df[(df['Phase'] == 'Wait') & (df['NODES'] == n)]
    mpi_row = df[(df['Phase'] == 'MPI Setup') & (df['NODES'] == n)]
    if not wait_row.empty and not mpi_row.empty:
        mean_sum = wait_row['Avg'].values[0] + mpi_row['Avg'].values[0]
    elif not wait_row.empty:
        mean_sum = wait_row['Avg'].values[0]
    elif not mpi_row.empty:
        mean_sum = mpi_row['Avg'].values[0]
    else:
        mean_sum = np.nan
    means_sum.append(mean_sum)
means_sum = np.array(means_sum, dtype=float)
ax.plot(node_order, means_sum, marker='o', label='Wait+MPI Setup', color=phase_colors['Wait+MPI Setup'], zorder=2)

ax.set_xticks(node_order)
ax.set_xlabel('NODES')
ax.set_ylabel('Time (s)')
ax.set_title(f'Execution Phases per Node ({scaling_type.capitalize()} Scaling)')
ax.legend(title="Phase")
ax.grid(axis='both', linestyle='--', alpha=0.7)
plt.savefig(f"times.{scaling_type}.png")
print(f"Plot salvato in times.{scaling_type}.png")
