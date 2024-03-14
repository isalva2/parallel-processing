#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {

    // instantiate variables
    int done = 0, n, myid, numprocs, i, rc;
    double mypi, pi, h, sum, x, a;
    double pi25dt = 3.141592653589793238462643;

    // check cli args
    if (argc >= 2)
    {
        // get number of steps for approximation
        n = atoi(argv[1]);
    }
    else
    {
        printf("Usage: mpirun -np <numprocs> %s <number of intervals>\n", argv[0]);
        exit(0);
    }

    // begin MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Broadcast from root (rank 0)
       the number of steps n */
    MPI_Bcast(
        &n, 1, MPI_INT, // broadcast the value of n
        0, MPI_COMM_WORLD // start at rank = 0 using global communicator
    );

    // begin computation
    h = 1.0 / (double) n; // get stepsize width
    sum = 0.0; // get sub-sum of process

    // begin compute using static interleaving scheduling
    for (i = myid + 1; i <= n; i += numprocs)
    {
        /* obtain midpoint location of x for
           rectangular approximation.
           Next get height of function f(x)
           and multiply by h to get rectangular
           approximation at x. */
        x = h * ((double) i-0.5);
        sum += 4.0/(1.0+x*x);
    }

    mypi += h*sum;

    // Reduce sub sums of individual processes
    MPI_Reduce(
        &mypi, // address of send buffer
        &pi, // address of receive buffer
        1, // int number of elements in receive
        MPI_DOUBLE, // datatype of elements of send buffer
        MPI_SUM, // Reduce operation (in this case sum up processes)
        0, // rank of root process
        MPI_COMM_WORLD // global communicator
    );

    // print results
    if (myid == 0)
    {
        printf("Computed with %d processes\n", numprocs);
        printf("\nMy pi:\t\t%.16f\n", pi);
        printf("Real pi:\t%.16f\n", pi25dt);
        printf("Error:\t\t%.16f", fabs(pi-pi25dt));
    }
    // close MPI enviornment
    MPI_Finalize();

}
