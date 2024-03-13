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
5. 