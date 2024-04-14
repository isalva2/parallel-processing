# README

This README file documents the compiling and execution of the [CUDA®](https://developer.nvidia.com/cuda-toolkit) program [`matrixNormCUDA.cu`](https://github.com/isalva2/parallel-processing/blob/main/CUDA/matrix-norm/matrixNormCUDA.cu)

Compiling `.cu` files requires the use of the [NVIDIA CUDA compiler driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) `nvcc`. For this program that looks like this:

```bash
$ nvcc matrixNormCUDA.cu
```

The program is designed to accept workload, optional block size, and optional random seed as command line arguments. If no block size is specified, the program will run with a default block size of 32.

```bash
$ ./a.out <matrix_dimension> [block size] [random seed]
```

For convenience, the include shell file `cuda-compile.sh` will compile and run an example. First change permissions using the command `$ chmod +x cuda-compile.sh` and run the file as an executable to get the following output.

```bash

```