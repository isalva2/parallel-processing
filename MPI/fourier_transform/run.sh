#!/bin/bash

# Compile models
mpicc -o a.out pp_convolution.c
mpicc -o b.out cc_convolution.c
mpicc -o c.out td_convolution.c

# Run on 4 processes (excluding td model)

echo "two-dimensional convolution summary"
echo "---------------------------------------"

echo; echo "Model 1"
echo "---------------------------------------"
mpirun -np 4 ./a.out

echo; echo "Model 2"
echo "---------------------------------------"
mpirun -np 4 ./b.out

echo; echo "Model 3"
echo "---------------------------------------"
mpirun -np 8 ./c.out
