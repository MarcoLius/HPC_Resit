#!/bin/bash

# Request resources:
#SBATCH -N 1		# number of compute nodes. 
#SBATCH -c 4		# number of CPU cores, one per thread, up to 128
#SBATCH -o part1.%J.out
#SBATCH -e part1.%J.err
#SBATCH --mem=1G	# memory required, up to 250G on standard nodes
#SBATCH --time=0:15:0	# time limit for job (format:  days-hours:minutes:seconds)

# Run in the 'shared' queue (job may share node with other jobs)
#SBATCH -p shared

# Modules necessary for job:
module purge
module load gcc/13.2

# clear existing outputs
rm part1.dat part1

# compile part1.c into part1
cc -fopenmp -O3 part1.c -Wall -march=native -o part1 -lm

# run part1 with 4 threads
OMP_NUM_THREADS=4 ./part1
