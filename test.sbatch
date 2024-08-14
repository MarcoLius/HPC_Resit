#!/bin/bash

# Request resources:
#SBATCH -N 1		# number of compute nodes. 
#SBATCH -n 4		# number of MPI ranks (1 per CPU core)
#SBATCH -o test.%J.out
#SBATCH -e test.%J.err
#SBATCH --mem=1G	# memory required per node, in units M, G or T
#SBATCH --time=0:15:0	# time limit for job (format:  days-hours:minutes:seconds)

# Run in the 'shared' queue (job may share node with other jobs)
#SBATCH -p shared 

# Modules necessary for job:
module purge
module load gcc/13.2 openmpi

# clear existing outputs
rm test.dat test

# compile part2.c into part1
mpicc -O3 test.c -Wall -march=native -o test -lm

# run part2 with 4 processes (from #SBATCH -n, above)
mpirun -np 4 ./test
