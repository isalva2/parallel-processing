#!/bin/bash

# CUDA compile
nvcc matrixNormCUDA.cu

# Execute
./a.out 10 5