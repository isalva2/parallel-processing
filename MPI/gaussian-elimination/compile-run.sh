#!/bin/bash

echo "$0"
echo

# Compile
mpicc -o gauss-mpi.out gauss-mpi.c

# Run
mpirun -np 4 ./gauss-mpi.out 10 2
