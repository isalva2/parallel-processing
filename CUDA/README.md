# README

This documentation describes the computational environment, compilation, runtime execution, and experimental results (i.e. printout verifications of accuracy) of the three [CUDA](https://developer.nvidia.com/cuda-zone#:~:text=CUDA%C2%AE%20is%20a%20parallel,harnessing%20the%20power%20of%20GPUs.) exercises.

## 1. Compute Environment
The compute environment for this exercise was hosted on [Chameleon Cloud](https://www.chameleoncloud.org/) on a single compute node with GPU capabilities. This physical hardware for this node consisted of two AMD [EPYC™ 7763](https://www.amd.com/en/products/cpu/amd-epyc-7763) CPUs and one NVIDIA® [A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/).

The compute node hosted a baremetal instance of [Ubuntu 20.04](https://releases.ubuntu.com/focal/) with NVIDIA® [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit) preinstalled. Using the command `nvcc --version` we can confirm that the system is running CUDA 12.2.

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jul_11_02:20:44_PDT_2023
Cuda compilation tools, release 12.2, V12.2.128
Build cuda_12.2.r12.2/compiler.33053471_0
```

Furthermore, using the CUDA utility `deviceQuery` located in the installation of `CUDA-12.2`, we can learn further details about the A100 GPU:

```bash
cc@chameleon-cloud:/cuda-12.2/extras/demo_suite/$ ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA A100-PCIE-40GB"
  CUDA Driver Version / Runtime Version          12.2 / 12.2
  CUDA Capability Major/Minor version number:    8.0
  Total amount of global memory:                 40339 MBytes (42298834944 bytes)
  (108) Multiprocessors, ( 64) CUDA Cores/MP:     6912 CUDA Cores
  GPU Max Clock rate:                            1410 MHz (1.41 GHz)
  Memory Clock rate:                             1215 Mhz
  Memory Bus Width:                              5120-bit
  L2 Cache Size:                                 41943040 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 151 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.2, CUDA Runtime Version = 12.2, NumDevs = 1, Device0 = NVIDIA A100-PCIE-40GB
Result = PASS
```

At 108 multiprocessors with 2,048 threads each, the theoretical concurrent threads we could run on this GPU is 221,184. Whether this can actually be implemented in practice is the subject of a later section.

## 2. Compiling and Execution

Compiling `.cu` files requires the use of the [NVIDIA CUDA compiler driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) `nvcc`. For the first source file `hello.cu`, compiling looks like this:

```bash
$ nvcc -o hello.out hello.cu
```

and results in the executable `hello.out`. The compiling process for the other source files are identical.

Running CUDA binaries is identical to executing compiled C source code. For the three `.cu` files it looks like this:

```bash
$ ./hello.out
$ ./vector_add.out
$ ./matrix_sum.out
```
## 3. Experimental Results

### Exercise 1: `hello.cu`

The first CUDA source file is a GPU "hello world!" example. Running the executable results in a hello from 4 threads in two thread blocks.

```bash
$ ./hello.out
Hello from block: 0, thread: 0
Hello from block: 0, thread: 1
Hello from block: 1, thread: 0
Hello from block: 1, thread: 1
```
### Exercise 2: `vector_add.cu`

The second exercise consisted of vector addition between vectors `A` and `B`, both 4,096 elements long. The kernel function `vadd()` was used to perform this operation in GPU, and in the source file the kernel was called in main as follows:

```cpp
vadd<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
```

At `DSIZE = 4096` and `block_size = 256`, the execution configuration argument `DSIZE + block_size - 1) / block_size` ensures that the correct grid and block size is used to parallelize the 4,096 elements of the vector addition operation.

At run time, the executable produces the following results:

```bash
$ ./vector_add.out
A[0] = 0.840188
B[0] = 0.394383
C[0] = 1.234571
```
And for good measure, a modified executable `vector_add_300.out` was used to validate the operation was correct for an element computed outside of the first thread block:

```bash
$ ./vector_add_300.out
A[300] = 0.761612
B[300] = 0.963505
C[300] = 1.725117
```

### Exercise 3: `matrix_sum.cu`

The third exercise consisted of row-wise and column-wise summation of a square matrix `A`. The kernel function for the row-wise summation looks like this