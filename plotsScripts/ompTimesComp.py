import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

if len(sys.argv) < 2:
    print("Usage: python ompTimesComp.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

# Prendi solo le colonne che servono
x = df['OMP_THREADS'].astype(int)

tot_time = df['Iter']
update = df['Update']
wait_mpi = df['Wait'] + df['MPIsetup']



plt.figure(figsize=(8, 6))
plt.plot(x, tot_time, marker='o', label='Total time (iterations)', color='#1f77b4', linestyle='-')
plt.plot(x, update, marker='o', label='Update', color='#d62728', linestyle='--')
plt.plot(x, wait_mpi, marker='o', label='Wait + MPI Setup', color='#ff7f0e', linestyle=':')

plt.xlabel('OMP Threads')
plt.ylabel('Time (s)')
plt.title('Execution Phases per Thread (omp scaling)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("times.omp.png")
print("Plot salvato in times.omp.png")