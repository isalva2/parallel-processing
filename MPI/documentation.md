# Message Passing Interface (MPI)

## About
Message Passing Interface (MPI) is a standardized message-passing API standard. MPI is a library specification, and relies on syntax and library routines to write portable message-passing routines in C, C++, and Fortran. MPI can be implemented on parallel computers, clusters, and heterogenous networks.

MPI is available in several open-source flavors, and this documentation will focus on [MPICH](https://www.mpich.org/about/overview/), an implementation of the MPI Standard developed by [Argonne National Laboratory (ANL)](https://www.anl.gov/).

## Installation Guide
I use a Mac, so I installed MPICH with [Homebrew ](https://brew.sh/) package manager using the command `$ brew install mpich`. This install was fairly painless compared to trying to install OpenMP or Pthreads.

For linux devices, an install guide from ANL is available [here](https://www.mpich.org/static/downloads/4.0.1/mpich-4.0.1-installguide.pdf).

## Basics
MPI has over [100 functions](https://www.mpich.org/static/docs/v3.0.x/www3/), but only 6 are needed to start writing MPI programs 🤩.

| Function | Purpose |
| :--- | --- |
|`MPI_Init`|Initiate an MPI program|
|`MPI_Finalize`|Terminate MPI program|
|`MPI_Comm_size`|Determine number of processes|
|`MPI_Comm_rank`|Determine the rank (ID) of the process|
|`MPI_Send`|Send a message|
|`MPI_Recv`|Receive a message|

In particular for `MPI_Send()` and `MPI_Recv()`:
### `MPI_Send()`
**Syntax**
```c
int MPI_Send(
    void *buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
)
```
**Parameters**
- `buf`: initial address of send buffer (choice)
- `count`: number of elements in send buffer (nonnegative integer)
- `datatype`: datatype of each send buffer element (handle)
- `tag`: message tag (integer)
- `comm`: communicator (handle)
On the initiation of an `MPI_Send()`, the source (root):
1. Allocates system space for the contents of the buffer using the `count` argument
2. Copies the contents of the buffer into system space
3. Uses `tag` and `dest` args to record the availability of a message for the destination processor
4. Returns to processing the user program

### `MPI_Recv()`
**Syntax**
```c
int MPI_Recv(
    void *buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm comm,
    MPI_Status *status
)
```
**Parameters**
- `buf`: initial address of receive buffer (choice) (OUT)
- `status`: status object (Status) (OUT)
- `count`: maximum number of elements in receive buffer (integer)
- `datatype`: datatype of each receive buffer element (handle)
- `tag`: message tag (integer)
- `comm`: communicator (handle)
On the initiation of a `MPI_Recv()` operation, the destination processor:
1. uses the `tag` and `source` argument to check the availability of a message from the source processor
2. If the message has been received, the message is copied ino the user's buffer; else it waits until the arrival of the message
3. Returns to processing the use program

## Hello World!
Here's an example of "Hello World!" using MPI and the below source code, `Hello-World-MPI.c`:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int npes, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    printf("From process %d out of %d, Hello World!\n", myrank, npes);
    MPI_Finalize();
    return 0;
}
```
This was confusing to me initially, so let's go line-by-line and explain:

1. `#include <mpi.h>` points us towards the MPI header file
2. `int npes, myrank;` are integer variables corresponding to the number of processors and rank (or ID) of each process, respectively.
3. `MPI_Init(&argc, &argv);` initializes MPI environment
4. `MPI_Comm_size(MPI_COMM_WORLD, &npes);`&dagger; obtains the number of processes in the MPI environment and and stores this value to the varaible `npes`
5. `MPI_Comm_rank(MPI_COMM_WORLD, &myrank);`&dagger; finds out the rank of the process and stores the rank in the variable `myrank`
6. `printf("From process %d out of %d, Hello World!\n", myrank, npes);` prints out a hello world statement from each process
7. `MPI_Finalize();` ends the MPI environment

[&dagger;]: Note that `MPI_Comm_size()` and `MPI_Comm_rank()` are obtained with respect to the **global** communicator `MPI_COMM_WORLD`. A communicator defines the group of processes that can communicate with one another, and as the name implies this communicator encompasses all processes in the MPI environment.

## Compiling with `MPICC`
Compiling an MPI program is similar to using standard compilers such as `gcc` and `clang`, but instead we make use of the wrapper `mpicc`:
```bash
$ mpicc -o Hello-World-MPI Hello-World-MPI.c
```
Finally, we can run an MPI program with `mpirun`
```bash
$ mpirun -np <npes> Hello-World-MPI
```
where `<npes>` specifies the number of processes at runtime.