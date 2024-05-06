#!/bin/bash

# Compile models
mpicc -o a.out model1_pp.c
mpicc -o b.out model2_cc.c
mpicc -o c.out model3_td.c

# Run on 4 processes (excluding td model)

echo "Two-dimensional Convolution Summary"
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

# RMSE verification
echo; echo "RMSE Summary"
echo "---------------------------------------"

gcc rmse_check_convolution.c
echo
./a.out "model1_out2"
./a.out "model2_out2"
./a.out "model3_out2"