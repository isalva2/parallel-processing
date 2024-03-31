#!/bin/bash

# Compile MPI program
mpicc -o gauss-mpi.out gauss-mpi.c

# Run example
mpirun -np 3 ./gauss-mpi.out 10 2
