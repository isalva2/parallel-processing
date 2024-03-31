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

At two CPUs per node, the compute environment for this exercise was able to utilize up to 80 cores and 160 threads for testing and validation. The compute node hosted a baremetal instance of [Ubuntu 20.04](https://releases.ubuntu.com/focal/) with [MPICH 4.20](https://www.mpich.org/2024/02/09/mpich-4-2-0-released/) preinstalled. The MPICH implementation is also well maintained, with an up-to-date [release page](https://www.mpich.org/downloads/) and [installation guide](https://www.mpich.org/downloads/).

## Compiling and Execution

To compile the parallel program [`gauss-mpi.c`](https://github.com/isalva2/parallel-processing/blob/main/MPI/gaussian-elimination/gauss-mpi.c), we follow the standard MPICH implementation that uses the `mpicc` command to link the program file to an executable.

```bash
$ mpicc -o gauss-mpi.out gauss-mpi.c
```

Upon compilation, the MPI program can be ran using the terminal command `mpicc` and specifying MPI and program parameters:

```bash
$ mpirun -np p ./gauss-mpi.out N s
```
where the `-np` flag allows us to specify the number of`p` processes used, `N` is the size of matrix `A`, and `s` is an optional random seed for reproducibility. For example, deterministically solving a matrix of size 5 with 3 processes would look like this:

```bash
$ mpirun -np 3 ./gauss-mpi.out 5 2
```
```
Random seed = 2

Matrix dimension N = 5.

Initializing...

A =
         1.03, 6166.23, 43380.29, 50255.05, 45372.79;
        17240.92, 23444.16, 4531.44, 8761.15, 3529.61;
        33502.70, 23500.74, 7007.32, 54720.72, 12054.02;
        60116.19, 56976.07, 3892.86, 24480.67, 20174.47;
        4294.87, 50266.38, 22432.89, 11662.35, 54525.87;

B = [28699.42; 2544.94; 1009.02; 56418.17; 26370.45]

Starting clock.
Computing with MPI using 2 processes
Stopped clock.
Sched time:     0.000029 seconds
Calc time:      0.000598 seconds
IO time:        0.000001 seconds

Elapsed time:   0.000628 seconds

X = [ 2.11; -1.35;  0.43; -1.10;  1.63]
--------------------------------------------

A =
         1.03, 6166.23, 43380.29, 50255.05, 45372.79;
         0.00, -103612416.00, -729088000.00, -844627968.00, -762576896.00;
         0.00,  1.76, 153350.56, 217559.44, 167574.53;
         0.00,  0.00,  0.00, -34206.12, 12769.54;
         0.00,  0.59, -0.01,  0.00, 67636.70;

B = [28699.42; -482348576.00; 98706.91; 58424.40; 109941.60]
```

In addition, for workloads smaller than `N <= 10`, the program output will display inputs `A` and `B` and there row-reduced counterparts following Gaussian elimination.

For convenience, the included shell script [`compile-run.sh`](https://github.com/isalva2/parallel-processing/blob/main/MPI/gaussian-elimination/compile.sh) will compile and run [`gauss-mpi.c`](https://github.com/isalva2/parallel-processing/blob/main/MPI/gaussian-elimination/gauss-mpi.c) as described above. Please make sure to change the file to executable with `chmod +x` before running using `bash compile-run.sh`.