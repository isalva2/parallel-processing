# Readme

This document summarizes the auxiliary tasks related to the implementation of Message Passing Interface (MPI) for the purposes of parallelizing and optimizing the Gaussian elimination algorithm. The compute environment for this program was hosted on [Chameleon Cloud](https://www.chameleoncloud.org/) in a Linux environment with an [MPICH](https://www.mpich.org/) installation. For more information on the design of the MPI program, pleas see the [DESIGN](https://github.com/isalva2/parallel-processing/blob/main/MPI/gaussian-elimination/DESIGN.md) file.

## Hardware and Compute Environment Summary

The compute environment was hosted on a Chameleon Cloud node with two Intel® [Xeon® Platinum 8380 CPUs](https://www.intel.com/content/www/us/en/products/sku/212287/intel-xeon-platinum-8380-processor-60m-cache-2-30-ghz/specifications.html) @ 2.30GHz. Intel notes the following specifications for the processor model:

| Specification               | Value        |
|-------------------------|--------------|
| Total Cores             | 40           |
| Total Threads           | 80           |
| Max Turbo Frequency     | 3.40 GHz     |
| Processor Base Frequency| 2.30 GHz     |
| Cache                   | 60 MB        |
| Intel® UPI Speed        | 11.2 GT/s    |
| Max # of UPI Links      | 3            |
| TDP                     | 270 W        |

At two CPUs per node, the compute environment for this exercise was able to utilize up to 80 cores and 160 threads for testing and validation. The compute node hosted a baremetal instance of [Ubuntu 20.04](https://releases.ubuntu.com/focal/) with [MPICH 4.20](https://www.mpich.org/2024/02/09/mpich-4-2-0-released/) preinstalled.

## Compiling and Execution

To compile the parallel program `gauss-mpi.c`, we follow the standard MPICH implementation that uses the `mpicc` command to link the program file to an executable.

```bash
$ mpicc -o gauss-mpi.out gauss-mpi.c
```

Upon compilation, the MPI program can be ran using the terminal command `mpicc` and specifying program parameters:

```bash
$ mpirun -np p ./gauss-mpi.out N s
```
where the `-np` flag allows us to specify the number of`p` processes used, `N` is the size of matrix `A`, and `s` is an optional random seed for reproducibility.


For convenience, the included shell script [`compile.sh`](https://github.com/isalva2/parallel-processing/blob/main/MPI/gaussian-elimination/compile.sh)