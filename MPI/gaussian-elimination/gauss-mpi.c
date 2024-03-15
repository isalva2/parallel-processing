/*
Isaac Salvador
CS566 Parallel Processing
Homework 3 MPI Programming
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 20000 /* Max value of N */
int N;            /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* MPI variables */
int numprocs, myid;
double start_time, stop_time; // get times from MPI routine

/* Prototype */
void gauss(); 

/* returns a seed for srand based on the time */
unsigned int time_seed()
{
    struct timeval t;
    struct timezone tzdummy;

    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char *argv[])
{
    int seed = 0; /* Random seed */

    /* Read command-line arguments */
    srand(time_seed()); /* Randomize */

    if (argc == 3)
    {
        seed = atoi(argv[2]);
        srand(seed);
        if (myid == 0)
        {
            printf("Random seed = %i\n", seed);
        }
    }
    if (argc >= 2)
    {
        N = atoi(argv[1]);
        if (N < 1 || N > MAXN)
        {
            if (myid == 0)
            {
                printf("N = %i is out of range.\n", N);
            }
            exit(0);
        }
    }
    else
    {
        if (myid == 0)
        {
            printf("Usage: %s <matrix_dimension> [random seed]\n", argv[0]);
        }
        exit(0);
    }

    /* Print parameters */
    if (myid == 0)
    {
        printf("\nMatrix dimension N = %i.\n", N);
    }
}


/* Initialize A and B (and X to 0.0s) */
void initialize_inputs()
{
    int row, col;

    printf("\nInitializing...\n");
    for (col = 0; col < N; col++)
    {
        for (row = 0; row < N; row++)
        {
            A[row][col] = (float)rand() / 32768.0;
        }
        B[col] = (float)rand() / 32768.0;
        X[col] = 0.0;
    }
}

/* Print input matrices */
void print_inputs()
{
    int row, col;

    if (N < 10)
    {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++)
        {
            for (col = 0; col < N; col++)
            {
                printf("%5.2f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < N; col++)
        {
            printf("%5.2f%s", B[col], (col < N - 1) ? "; " : "]\n");
        }
    }
}

void print_X()
{
    int row;

    if (N < 100)
    {
        printf("\nX = [");
        for (row = 0; row < N; row++)
        {
            printf("%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n");
        }
    }
}

/* Helper Functions */
// print process rank
void shout()
{
    printf("my rank: %d\n", myid);
}

int main(int argc, char *argv[])
{
    // Initialize MPI environment and num processes and rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank( MPI_COMM_WORLD , &myid);

    // Initialize parameters, everyone gets N
    parameters(argc, argv);

    // Start Clock
    if (myid == 0)
    {
        initialize_inputs();
        printf("Starting clock.\n");
        start_time = MPI_Wtime();
    }

    // Barrier after starting clock to sync up processes
    MPI_Barrier(MPI_COMM_WORLD);

    shout();
    printf("my N: %d, %f\n", N, start_time);

    // Stop Clock and runtime logging
    if (myid == 0)
    {
        stop_time = MPI_Wtime();
        printf("Elapsed time = %f seconds\n", stop_time-start_time);
    }
    // Exit MPI enviornment
    MPI_Finalize();
    exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss_source()
{
    int norm, row, col; /* Normalization row, and zeroing
                         * element row and col */
    float multiplier;

    printf("Computing Serially.\n");

    /* Gaussian elimination */
    for (norm = 0; norm < N - 1; norm++)
    {
        for (row = norm + 1; row < N; row++)
        {
            multiplier = A[row][norm] / A[norm][norm];
            for (col = norm; col < N; col++)
            {
                A[row][col] -= A[norm][col] * multiplier;
            }
            B[row] -= B[norm] * multiplier;
        }
    }
    /* (Diagonal elements are not normalized to 1.  This is treated in back
     * substitution.)
     */

    /* Back substitution */
    for (row = N - 1; row >= 0; row--)
    {
        X[row] = B[row];
        for (col = N - 1; col > row; col--)
        {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
}
