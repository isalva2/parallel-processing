/*
Isaac Salvador
CS566 Parallel Processing
Homework 3 MPI Programming
*/

#include <stdio.h> // Headers
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

#pragma region // Global variables and Gauss prototype

/* Program Parameters */
#define MAXN 20000 /* Max value of N */
int N;             /* Matrix size */

/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* Prototype */
void gauss_mpi();

/* MPI variables */
int numprocs, myid;

/* Recording */
double start_time;
double start_sched, stop_sched;
double stop_calc;
double stop_time;

#pragma endregion

#pragma region // inputs and initialization

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

    if (N <= 10)
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

#pragma endregion

int main(int argc, char *argv[])
{   
    // Initialize MPI environment and num processes and rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Initialize parameters, everyone gets N
    parameters(argc, argv);

    // Initialize inputs and Start Clock
    if (myid == 0)
    {
        initialize_inputs();
        print_inputs();
        printf("\nStarting clock.\n");
        start_time = MPI_Wtime();
    }

    // Compute Gaussian elimination
    gauss_mpi();

    // Stop Clock and runtime logging
    if (myid == 0)
    {
        stop_time = MPI_Wtime();
        printf("Stopped clock.\n");
        print_inputs();
        print_X();

        // Print run time report
        double t_sched = stop_sched - start_sched;
        double t_calc = stop_calc - stop_sched;
        double t_total = stop_time - start_time;
        double t_IO = t_total - t_sched - t_calc;
        printf("Sched time:\t%f seconds\n", t_sched);
        printf("Calc time:\t%f seconds\n", t_calc);
        printf("IO time:\t%f seconds\n", t_IO);
        printf("\nElapsed time:\t%f seconds\n", stop_time - start_time);
        printf("--------------------------------------------\n");
    }

    // Exit MPI environment
    MPI_Finalize();
    exit(0);
}

void gauss_mpi()
{
    // Algorithm variables
    int norm, row, col, proc;
    float multiplier;

    // Declare status object for MPI_Recv()
    MPI_Status status;

    // Start recording scheduling time:
    if (myid == 0)
    {
        start_sched = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Begin static interleaved scheduling by root process
    
    // Root process Does the scheduling
    if (myid == 0) 
    {
        for (row = 1; row < N - 1; row ++)
        {
            // Static assignment of rows to each worker process
            proc = row % numprocs;
            if (proc != 0)
            {
                // Send rows of A and corresponding value of B
                MPI_Send(&A[row + 1][0], N, MPI_FLOAT, proc, 0, MPI_COMM_WORLD);
                MPI_Send(&B[row + 1], 1, MPI_FLOAT, proc, 1, MPI_COMM_WORLD);
            }
            
        }
    }
    // Worker processes receive rows
    else
    {
        for(row = 1; row < N - 1; row++)
        {   
            if (myid == row % numprocs)
            {
                // Corresponding send and receives
                MPI_Recv(&A[row + 1][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&B[row + 1], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
            }
        }
    }

    // Broadcast initial 0-th norm row
    MPI_Bcast(&A[0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B[0], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Stop recording scheduling time
    if (myid == 0)
    {
        stop_sched = MPI_Wtime();
    }
    
    // Begin Gaussian elimination step
    for (norm = 0; norm < N - 1; norm++)
    {
        // New main calc using modulo
        for (row = norm + 1; row < N; row ++)
        {
            if (myid == (row - 1 + numprocs) % numprocs)
            {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++)
                {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }
        }
        
        // norm-th proc broadcasts next, completed, norm + 1 row for next iteration
        proc = norm % numprocs;
        MPI_Bcast(&A[norm+1][0], N, MPI_FLOAT, proc, MPI_COMM_WORLD);
        MPI_Bcast(&B[norm+1], 1, MPI_FLOAT, proc, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Back substitution computer by root
    if (myid == 0)
    {
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

    MPI_Barrier(MPI_COMM_WORLD);

    // Stop recording Gaussian elimination step
    if (myid == 0)
    {
        stop_calc = MPI_Wtime();
    }
}
