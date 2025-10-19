#!/bin/bash

echo "Threads scaling: single node with multiple threads"

NODES=1
N_TASKS_PER_NODE=1
TOTAL_TASKS=1
N_STEPS=750
GRID_SIZE_X=10000
GRID_SIZE_Y=10000
EXECUTABLE=${1:-./stencil5}  #CLI accepting one optional argument: executable path. Example: bash scripts/omp_scaling.sh ./stencil5.02

for OMP_THREADS in 1 2 4 8 16 32 56 84 112 ; do
    JOB_NAME="thread_scaling_${OMP_THREADS}_threads"

    sbatch --export=ALL,GRID_SIZE_X=${GRID_SIZE_X},GRID_SIZE_Y=${GRID_SIZE_Y},N_STEPS=${N_STEPS},OMP_THREADS=${OMP_THREADS},JOB_NAME=${JOB_NAME},TOTAL_TASKS=${TOTAL_TASKS} --nodes=${NODES} --ntasks-per-node=${N_TASKS_PER_NODE} --cpus-per-task=${OMP_THREADS} --job-name=${JOB_NAME} scripts/go_dcgp.sh $EXECUTABLE
done

echo "All threads scaling jobs submitted"
