#!/usr/bin/env python3
# OpenMP strong-scaling plots with ideal (theoretical) curves

""" ================================================================
    = CLI: python3 strongS_weakS.py [strong|weak] input_csv_file
    ================================================================ """


import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend so plots can be saved to PNG on headless systems
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# === Config ==============================================================
# take two arguments from CLI: oe for weak or strong scaling (changes the theoretical curves)
import sys
if sys.argv[1] not in ['strong', 'weak']:
    print("Usage: python3 strong-weak.py [strong|weak]")
    sys.exit(1)

scaling_type = sys.argv[1]  

# second argument the name of the input csv file, indipendently from the scalability test type
if len(sys.argv) != 3:
    print("Usage: python3 strong-weak.py [strong|weak] input_csv_file")
    sys.exit(1)
RAW_CSV   = Path(sys.argv[2])            # input originale
PROC_CSV  = Path(f"{scaling_type}.csv") # output
FIG_TIME  = Path(f"{scaling_type}T.png")
FIG_SPEED = Path(f"{scaling_type}SU.png")
FIG_EFF   = Path(f"{scaling_type}E.png")

NODE_COL    = "NODES"          # colonna con numero di nodi
TIME_COL    = "Iter"          # metrica di tempo per iterazione (usa "Total" se preferisci)
#=====================================================================

# --- Carica e seleziona colonne
raw = pd.read_csv(RAW_CSV)
df = raw[[NODE_COL, TIME_COL]].copy()
df[NODE_COL] = pd.to_numeric(df[NODE_COL], errors="coerce")
df[TIME_COL]    = pd.to_numeric(df[TIME_COL], errors="coerce")
df = df.dropna().sort_values(NODE_COL)

# baseline T1 (preferibilmente a 1 nodo)
if (df[NODE_COL] == 1).any():
    T1 = float(df.loc[df[NODE_COL] == 1, TIME_COL].iloc[0])
    base_nodes = 1
else:
    # se manca 1 nodo, usa il primo valore come baseline (p_min)
    base_nodes = int(df[NODE_COL].iloc[0])
    T1 = float(df[TIME_COL].iloc[0])

# --- Metriche: speedup ed efficienza
if scaling_type == 'strong':
    df["speedup"]    = T1 / df[TIME_COL]
    df["efficiency"] = df["speedup"] / df[NODE_COL]
else:
    df["speedup"] = T1 / (df[TIME_COL] / df[NODE_COL])  # in weak scaling, speedup is equal to number of nodes
    df["efficiency"] = df["speedup"] / df[NODE_COL]

# --- Salva CSV “pulito”
out = df[[NODE_COL, TIME_COL, "speedup", "efficiency"]].rename(
    columns={NODE_COL: "nodes", TIME_COL: "iter_time"}
)
out.to_csv(PROC_CSV, index=False)

# --- Curve teoriche
nodes = out["nodes"].to_numpy()
if scaling_type == 'strong':
    # strong scaling: tempo ideale T(p) = T1/p
    ideal_time = T1 / nodes
    ideal_speedup = nodes
    ideal_efficiency = np.ones_like(nodes)
else:
    # weak scaling: tempo ideale T(p) = T1 (costante)
    ideal_time = np.full_like(nodes, T1, dtype=float)
    ideal_speedup = nodes  # in weak scaling, speedup is equal to number of nodes
    ideal_efficiency = np.ones_like(nodes)

# --- Plots
# === Grafico 1: Tempo per iterazione vs Nodi ==================================
plt.figure(figsize=(8, 6))
plt.plot(out["nodes"], out["iter_time"], 'o-', label="Measured", color='blue')
plt.plot(nodes, ideal_time, 'r--', label="Ideal", color='orange')
plt.xlabel("Number of Nodes")
plt.ylabel("Time for Iterations")
plt.title(f"{scaling_type.capitalize()} Scaling: Time for Iterations")
plt.legend()
plt.grid(True)
if scaling_type == 'weak':
    plt.ylim(bottom=0, top=T1 * 1.20)  # in weak scaling, time should remain constant
plt.savefig(FIG_TIME)
plt.close() 

# === Grafico 2: Speedup vs Nodi ==============================================
plt.figure(figsize=(8, 6))
plt.plot(out["nodes"], out["speedup"], 'o-', label="Measured", color='blue')
plt.plot(nodes, ideal_speedup, 'r--', label="Ideal", color='orange')
plt.xlabel("Number of Nodes")
plt.ylabel("Speedup")
plt.title(f"{scaling_type.capitalize()} Scaling: Speedup")
plt.legend()
plt.grid(True)
plt.savefig(FIG_SPEED)
plt.close() 

# === Grafico 3: Efficienza vs Nodi ============================================
plt.figure(figsize=(8, 6))
plt.plot(out["nodes"], out["efficiency"], 'o-', label="Measured", color='blue')
plt.plot(nodes, ideal_efficiency, 'r--', label="Ideal", color='orange')
plt.xlabel("Number of Nodes")
plt.ylabel("Efficiency")
plt.title(f"{scaling_type.capitalize()} Scaling: Efficiency")
plt.legend()
plt.grid(True)
plt.ylim(bottom=0, top=1.20)
plt.savefig(FIG_EFF)
plt.close() 

# ========================================================================
print (f"Processed data saved to {PROC_CSV}")
print (f"Plots saved to {FIG_TIME}, {FIG_SPEED}, {FIG_EFF}")

# ========================================================================
