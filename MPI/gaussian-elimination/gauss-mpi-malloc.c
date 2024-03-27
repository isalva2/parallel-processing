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
#include <limits.h>

#pragma region // MPI Helper functions

// MPI Variables
int numprocs, myid;
double start_time, stop_time; // get times from MPI routine

// Global Guassian elimination variables
int norm, row, col, proc;

// Helper functions
int calc_A_size(int norm, int proc);
int zero_array(const float *array, int size);

// MPI Prototypes
void root_send(int norm);
void root_recv();
void worker_send();
void worker_recv();

#pragma endregion

#pragma region // Global variables and helper functions

/* Program Parameters */
#define MAXN 20000 /* Max value of N */
int N;             /* Matrix size */

/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* Prototype */
void gauss_mpi();

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
        print_X();
        printf("\nElapsed time = %f seconds\n", stop_time - start_time);
        printf("--------------------------------------------\n");
    }

    // Exit MPI environment
    MPI_Finalize();
    exit(0);
}

void gauss_mpi()
{
    int norm, row, col, proc;
    float multiplier;
    
    // synch up processes
    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0)
    {
        printf("Computing in parallel with %d processes.\n", numprocs);
    }

    /* NEW STUFF GOES HERE */

    // Skeleton code for worker/root processes
    for (norm = 0; norm < N - 1; norm++)
    {
        MPI_Bcast(&A[norm][0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[norm], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (myid ==0)
        {
            root_send(norm);
        }
        else
        {
            // void;
        }
    }


    // // Barrier before back substitution
    // MPI_Barrier(MPI_COMM_WORLD);

    // // Back substitution computer by root
    // if (myid == 0)
    // {
    //     for (row = N - 1; row >= 0; row--)
    //     {
    //         X[row] = B[row];
    //         for (col = N - 1; col > row; col--)
    //         {
    //             X[row] -= A[row][col] * X[col];
    //         }
    //         X[row] /= A[row][row];
    //     }
    // }
}

#pragma region // MPI Helper Functions

// Definitions

// This returns the number of elements of A that need to be sent.

/* THIS WILL MOST LIKELY CAUSE PROBLEMS */
int calc_A_size(int norm, int proc)
{
    int A_size = ((int)ceil((N-(norm+1+proc)))/numprocs+1)*(N);
    return A_size;
}

// Check if array is sending zeroes
int zero_array(const float *array, int size);

// This function sends to numprocs-1 worker processes the necessary rows of A and B to each worker process. This is the complimentary function to worker_recv(). For each worker process the root process will send a MPI_Send() of a float buffer to send floats from A and B, and an int buffer to send the size of the float buffer.
void root_send(int norm)
{   
    // Root sends to each worker process
    for (proc = 1; proc < numprocs; proc++)
    {
        // A_size is dependent on what row the process works on, how many processes there are, and at what norm the gaussian elimination step is on.

        // The number of elements of A that need to be sent
        int A_size = calc_A_size(norm, proc);
        int B_size = N;

        // Get the size of the MPI Send buffer, sends elements of A and N elements of B
        int buffer_size = A_size + B_size;

        // Declare buffer
        // float root_buffer[buffer_size];

        // Try malloc buffer
        float *root_buffer = (float *)malloc(buffer_size * sizeof(float));

        // Initalize buffer to zero for now
        // for (int i = 0; i < buffer_size; i++)
        // {
        //     root_buffer[i] = 0.0;
        // }

        // initialize values of Root's A and B to buffer. B starts at displacement of size A_size
        int A_index = 0, B_index = 0;
        for (row = norm + 1 + proc; row < N; row += numprocs)
        {
            // Lets try sending the entire row 
            for (col = row; col < N; col ++)
            {
                root_buffer[A_index] = A[row][col]; // Store value of A in buffer
                A_index++;
            }
            root_buffer[A_size+B_index] = B[row];
            B_index++;

        }

        // Make sure that erroneous scheduling (where A_size = 0) does not happen
        if (A_size)
        {
            printf("At norm = %d, root %d sent A buffer size %d to proc %d\n", norm, myid, A_size, proc);
        }
        

        // // quick check
        // printf("\nBuffer to proc %d from root %d at norm = %d\n", proc, myid, norm);
        // for (int i = 0; i < buffer_size; i++)
        // {
        //     printf("%8.2f\t", root_buffer[i]);
        // }
        // printf("\n");

        free(root_buffer);

        // // buffer filled with values, now send to proc
        // MPI_Send( const void* buf , MPI_Count count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm);
        // MPI_Send(&root_buffer[0], buffer_size, MPI_FLOAT, proc, 0, MPI_COMM_WORLD);

        // // I HOPE THIS WORKS ONLY numprocs*2 SEND/RECV'S PER NORM ITERATION

    }
}




#pragma endregion
