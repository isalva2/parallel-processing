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

For convenience, the include shell file `cuda-compile.sh` will compile and run an example. First change permissions using the command `$ chmod +x cuda-compile.sh` and run the file as an executable to get the following output. If all goes well the output or similar will be displayed.

```bash
Random seed = 2

Initializing...

---------------------------------------------
Matrix size N = 6

Starting clock.

45939.19	53062.95	 5819.30	 7961.26	22826.63	27653.70
45862.45	 4350.56	38501.24	42137.44	64920.18	19380.19
17782.32	 4564.97	62235.57	25046.24	21149.13	64773.52
55215.53	18042.99	39100.52	36465.56	55568.50	12099.50
26048.58	58311.50	55737.10	20822.74	 6748.12	 2942.60
42270.14	52687.30	56005.55	48089.44	60648.57	13296.18

Kernel Launch Parameters:
-------------------------
Blocks:		3
Threads/Block:	2
Total threads:	6

Stopped clock.

Runtime = 2401.84 ms.

    0.55	    0.91	   -1.97	   -1.63	   -0.70	    0.22
    0.55	   -1.18	   -0.23	    0.89	    1.17	   -0.20
   -1.64	   -1.17	    1.03	   -0.37	   -0.78	    2.07
    1.28	   -0.59	   -0.20	    0.47	    0.75	   -0.56
   -1.00	    1.13	    0.68	   -0.68	   -1.42	   -1.02
    0.27	    0.89	    0.70	    1.32	    0.98	   -0.50

---------------------------------------------
```