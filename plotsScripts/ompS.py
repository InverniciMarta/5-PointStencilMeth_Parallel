#!/usr/bin/env python3
# OpenMP strong-scaling plots with ideal (theoretical) curves

""" ================================================================
    = CLI: python3 ompS.py input_csv_file                          =
    ================================================================ """

import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend so plots can be saved to PNG on headless systems
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# === Config ==============================================================
RAW_CSV   = Path(sys.argv[1]) 
if len(sys.argv) != 2:
    print("Usage: python3 omp_plots.py input_csv_file")
    sys.exit(1)
RAW_CSV   = Path(sys.argv[1])            # input originale
#all to comment if you want to compare different executions apart from FIG_TIME
PROC_CSV  = Path("omp.csv") # output con metriche 
FIG_TIME  = Path("ompT.png") #("ompT.orfeo01-02.png") 
FIG_SPEED = Path("ompSU.png")
FIG_EFF   = Path("ompE.png")

THREADS_COL = "OMP_THREADS"   # colonna con numero di thread
TIME_COL    = "Iter"          # metrica di tempo per iterazione (usa "Total" se preferisci)
TIME_IS_SECONDS = False       # metti True se TIME_COL è in secondi
# ========================================================================


# --- Carica e seleziona colonne (supporta anche 'exec' se presente)
raw = pd.read_csv(RAW_CSV)
has_exec = 'exec' in raw.columns
if has_exec:
    df = raw[['exec', THREADS_COL, TIME_COL]].copy()
else:
    df = raw[[THREADS_COL, TIME_COL]].copy()
df[THREADS_COL] = pd.to_numeric(df[THREADS_COL], errors="coerce")
df[TIME_COL]    = pd.to_numeric(df[TIME_COL], errors="coerce")
df = df.dropna().sort_values([THREADS_COL] if not has_exec else ['exec', THREADS_COL])

# --- Tempo in millisecondi (se necessario)
iter_time_ms = df[TIME_COL] * (1000.0 if TIME_IS_SECONDS else 1.0)
df["iter_time_ms"] = iter_time_ms.values

# --- Calcola metriche per ogni exec (o globalmente se non c'è)
results = []
group_cols = ['exec'] if has_exec else [None]
grouped = df.groupby('exec') if has_exec else [(None, df)]
for exec_name, group in grouped:
    group = group.copy()
    # Baseline T1 (preferibilmente a 1 thread)
    if (group[THREADS_COL] == 1).any():
        T1 = float(group.loc[group[THREADS_COL] == 1, TIME_COL].iloc[0])
        base_threads = 1
    else:
        base_threads = int(group[THREADS_COL].iloc[0])
        T1 = float(group[TIME_COL].iloc[0])
    group["speedup"]    = T1 / group[TIME_COL]
    group["efficiency"] = group["speedup"] / group[THREADS_COL]
    group["base_threads"] = base_threads
    group["T1"] = T1
    group["exec"] = exec_name if has_exec else 'default'
    results.append(group)

df_out = pd.concat(results)

# --- Salva CSV “pulito”
out = df_out[["exec", THREADS_COL, "iter_time_ms", "speedup", "efficiency"]].rename(
    columns={THREADS_COL: "threads"}
)
out.to_csv(PROC_CSV, index=False)

# --- Curve teoriche (unica, calcolata sulla baseline minima di tutti gli exec)
threads_all = sorted(out["threads"].unique())
min_T1 = df_out.groupby('exec')["T1"].min().min() if has_exec else df_out["T1"].min()
T_theory = np.array([min_T1 / t for t in threads_all])
S_theory = np.array(threads_all, dtype=float)
E_theory = np.ones_like(threads_all, dtype=float)

# === Grafico 1: Tempo per iterazione (ms) vs Threads =====================
plt.figure(figsize=(8, 6))
for exec_name, group in out.groupby("exec"):
    plt.plot(group["threads"], group["iter_time_ms"], marker="o", label=f"Measured({exec_name})", color='blue') #comment blue if...
plt.plot(threads_all, T_theory, linestyle="--", color="orange", label="Ideal")#black for comparison updates by rows vs blocks
plt.xlabel("Threads")
plt.ylabel("Time for Iterations")
plt.title("OpenMP Scaling: Time for Iterations")#("OpenMP Scaling Time: update by rows(1), update by blocks(2)")
plt.grid(True)
plt.legend() 
plt.savefig(FIG_TIME)
plt.close()

# === Grafico 2: Speedup vs Threads ======================================
plt.figure(figsize=(8, 6))
for exec_name, group in out.groupby("exec"):
    plt.plot(group["threads"], group["speedup"], marker="o", label=f"Measured({exec_name})", color='blue')
plt.plot(threads_all, S_theory, linestyle="--", color="orange", label="Ideal (p)")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("OpenMP Scaling: Speedup")
plt.grid(True)
plt.legend()
plt.ylim(bottom=0)
plt.savefig(FIG_SPEED)
plt.close()

# === Grafico 3: Efficienza vs Threads ===================================
plt.figure(figsize=(8, 6))
for exec_name, group in out.groupby("exec"):
    plt.plot(group["threads"], group["efficiency"], marker="o", label=f"Measured({exec_name})", color='blue')
plt.plot(threads_all, E_theory, linestyle="--", color="orange", label="Ideal (=1)")
plt.xlabel("Threads")
plt.ylabel("Efficiency")
plt.title("OpenMP Scaling: Efficiency")
plt.grid(True)
plt.legend()
plt.ylim(bottom=0, top=1.20)
plt.savefig(FIG_EFF) 
plt.close()

print(f"[OK] Saved: {PROC_CSV}, {FIG_TIME}, {FIG_SPEED}, {FIG_EFF}")
