
# HPC 5-Stencil Computation of 2D Heat Equation
This repository contains an advanced High-Performance Computing (HPC) implementation of a five-dimensional stencil computation, leveraging MPI for distributed memory parallelism and OpenMP for shared memory parallelism. The code is designed for efficient stencil updates on multidimensional grids, a common pattern in scientific simulations. This project was developed as part of the HPC 2024/2025 course at the University of Trieste.

## PROJECT TREE

```
HPCStencil/
├─ CSVs/            # CSV files with time statistics
│  └─ ..            
├─ plots/           # Plots of scaling time, speedup, efficiency and time components
│  └─ ..            
├─ plotScripts/     # Python scripts for data visualization
│  └─ ..            
├─ scripts/         # Bash scripts with embedded sbatch directives for Slurm job submission and experiment automation (adapted for Leonardo @ Cineca)
├─ assignment/      # Assignment slides
├─ slides/          # Final presentation slides
├─ stencil5.c      # Src file
├─ stencil5_H.h      # Header file 
├─ Makefile               
└─ README.md        

```

## FEATURES
* MPI for distributed memory parallelism.
* OpenMP for shared memory parallelism.
* Time profiling for performance analysis.
* Configurable behaviour at runtime via command-line arguments and environment variables.

## COMPILATION AND EXECUTION
To compile the code, use the provided Makefile from the project root directory:
```
make
```

To run the program, use `mpirun` or `mpiexec` with the desired number of processes. You can specify the value for a lot of command-line arguments, to customize the execution. To print all the available options the flag `-h` can be used. 
```
mpirun -np <num_processes> ./5Dstencil [options]
``` 

To delete the compiled executable and object files, use:
```
make clean
```

## LAUNCH SCRIPTS USAGE (Leonardo @ Cineca)

The `scripts/` directory contains Bash scripts with embedded Slurm (`sbatch`) directives, specifically adapted for the Leonardo supercomputer at Cineca. These scripts automate compilation, job submission, and scaling experiments. Each script is compatible with the partitioning and resource management policies of Leonardo.

**Available scripts:**
- `compile.sh`: Submits a compilation job to the cluster (contains `#SBATCH` directives).
- `omp_scaling.sh`: Automates OpenMP thread-scaling experiments by submitting multiple jobs with varying thread counts (1, 2, 4, 8, 16, 32, 56, 84, 112 threads).
- `strong_scaling.sh`: Automates strong-scaling experiments across multiple nodes (1, 2, 4, 8, 16 nodes) with fixed problem size.
- `weak_scaling.sh`: Automates weak-scaling experiments across multiple nodes (1, 2, 4, 8, 16 nodes) with proportionally increasing problem size.

**Usage examples:**


To compile the code via Slurm job submission from root directory:
```bash
sbatch scripts/compile.sh
```

To run scaling experiments from the root directory (these scripts submit multiple jobs internally via `sbatch`):
```bash
bash scripts/omp_scaling.sh
bash scripts/strong_scaling.sh
bash scripts/weak_scaling.sh
```

All scaling scripts accept an optional argument to specify a custom executable path:
```bash
bash scripts/omp_scaling.sh ./5Dstencil.02
bash scripts/strong_scaling.sh ./custom_executable
```

To override Slurm environment variables (e.g., partition) before running a script:
```bash
export SBATCH_PARTITION=partition_name
bash scripts/strong_scaling.sh
```

**Key points:**
- Resource requests (nodes, tasks, CPUs, memory) are set to match Leonardo's configuration; you may need to adjust these for other clusters.
- Scripts automate sweeps over thread counts, node counts, and grid sizes for comprehensive performance studies.
- Output files (CSV, logs) are written to the working directory (project root).

**Note:**
If you intend to use these scripts on a different HPC system, review and modify the `#SBATCH` directives in `go_dcgp.sh` and `compile.sh` to match the target cluster's partition names, resource limits, and environment modules. For Leonardo, the scripts are ready-to-use and tested for the DCGP partition and standard job policies.

For further details on the Leonardo system, see: [Cineca Leonardo documentation](https://leonardo-supercomputer.cineca.eu/))


## DATA GRID ACCESS STYLE
You can control how the data grid is scanned for memory allocation and updates via the following CLI parameter:
* `-l: 1` to enable block scanning of the grid (default 0).

This option can help improve performance by enhancing cache locality during memory allocation and update operations. When enabled, the grid will be processed in smaller blocks rather than as a whole, which can lead to better utilization of the CPU cache.

Would be ideal to experiment with different block sizes to find the optimal configuration for your specific hardware and problem size. Unfortunately, this code does not currently support changing the block size directly via a CLI parameter, as it is hardcoded in the implementation. We leave it as a future enhancement.




## GRID PRINTING FOR DEBUGGING AND VISUALIZATION
You can control grid printing for debugging or visualization reasons via the following environment variable and CLI parameters. 

Default is no grid printing at all. 

Notice that this operation can produce a large amount of output and be time expensive, especially for big grids or many time steps, which may slow down the execution significantly. For these reasons, grid printing is mainly intended for small grid sizes: for debugging should be human-readable sizes (e.g. <= 20x20). For visualization purposes, consider using moderate grid sizes that balance detail and performance.


CLI Options:
* `D: 1` to enable grid printing for debugging (default 0)
* `P: 1` to enable full grid printing at the end of the simulation (default 0)
* `d: 1` to enable halo printing (default 0)


Environment variable:
```
export GRID_PRINT_MODE=<mode> 
```
sets the mode of grid printing. Possible values are:

* `Debug`: to print at each step with halos for debugging
* `GridEachStep`: to print grid at each step without halos for visualization
* `GridFinal`: to print only final grid without halos for visualization

To go back to default behavior:
```
unset GRID_PRINT_MODE
```



## TIME STATISTICS
You can enable timing statistics output by setting the following environment variable before running the program:

```
export TIME_STATISTICS=1
```

to enable both time statistics output as csv files and stdout prints.

But the executable behavior can be customized by fine-tuning the following CLI Options:
* `-T 1` to enable verbose timing statistics output to stdout
* `-t 1` to enable timing statistics output as csv files

When either CLI `-t 1` or environment variable `TIME_STATISTICS=1`are enabled, the program will generate CSV files containing detailed time information for various parts of the computation. These files are written to the executable's working directory and can be used for performance analysis and optimization. This CSV files are the expected input for the plotting scripts in plotScripts/:

* `slowest_rank_times.csv` contains one row per run; the first line is a CSV header. Columns include run identifiers (timestamp, number of nodes, number of processes, number of threads, ..), values of problem parameters (e.g., grid size, time step, ..) and times measured per code regions (e.g., initialization, communications set up, update, ..).   

* `stats_avg_min_max.csv` contains, for each run, one row per profiled code region (n rows for n regions, with an upper header). Columns provide aggregated statistics across the tasks of that run — average, minimum and maximum time — together with run identifiers (timestamp, number of nodes, processes and threads, problem parameters values, ...) and values of problem parameters (e.g., grid size, time step, ..). 

## DATA VISUALIZATION SCRIPTS
Python scripts for data visualization are provided in the `plotScripts/` directory. To use these scripts, ensure you have the required Python libraries installed, such as Matplotlib and Pandas.

Run the scripts by navigating to the `plotScripts/`: 

* `ompS.py` generates OpenMP strong-scaling plots. It expects a CSV with the same structure as `slowest_rank_times.csv` containing time data from runs that vary only the number of OpenMP threads (do not mix MPI/node-scaling experiments).

Usage:
```
python ompS.py </path/to/csv/files>
```


* `strongS_weakS.py` plots strong- and weak-scaling results versus number of nodes. It expects a CSV with the same structure as `slowest_rank_times.csv` produced by runs that are exclusively strong-scaling or weak-scaling (do not mix types) over nodes.

Usage:
```
python strongS_weakS.py [strong|weak] </path/to/csv/files>
```

* `ompTimesComp.py` compares MPI(comunication) and OpenMP(computation) time statistics for strong-scaling OpenMP runs. It expects a CSV with the same structure as `slowest_rank_times.csv` containing time data from runs that vary only the number of OpenMP threads (do not mix MPI/node-scaling experiments).

Usage:
```
python ompTimesComp.py </path/to/csv/files>
```

* `strong_weakTimeComp.py` compares MPI(communication) and OpenMP(computation) time statistics for either strong-scaling or weak-scaling runs over nodes. It expects a CSV with the same structure as `stats_avg_min_max.csv`, produced by runs that are exclusively strong-scaling or weak-scaling (do not mix types) over nodes.

Usage:
```
python strong_weakTimeComp.py [strong|weak] </path/to/csv/files>
```

For each command above,`[strong|weak]` selects the plot type.


Once again, before running one of the python scripts, ensure the input csv contains only one scaling type: EITHER OpenMP-thread scaling (omp) OR node scaling (nodes) with no mixing strong or weak type tests. Mixing will produce incorrect or misleading plots.

You can run the script from its directory by simply passing the CSV path as an argument; generated plot files will be then saved to the same directory as the plotting script (plotScripts/).









