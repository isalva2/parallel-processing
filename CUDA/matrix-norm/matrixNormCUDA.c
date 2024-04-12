/*
Matrix Normalization using CUDA
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

/* Program parameters */
#define MAXN 6000
int N;

/* Declare host data */
float *h_A, *h_B;

/* Prototype matrix normalization */
void matrix_norm();

#pragma region // Helper functions

/* Returns a seed for srand based on the time */
unsigned int time_seed()
{
    struct timeval t;
    struct timezone tzdummy;

    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

/* Set program parameters */
void parameters(int argc, char **argv)
{
    int seed = 0;

    srand(time_seed()); /* Randomize */

    if (argc == 3)
    {
        seed = atoi(argv[2]);
        srand(seed);
        printf("Random seed = %i\n", seed);
    }
    if (argc >= 2)
    {
        N = atoi(argv[1]);
        if (N < 1 || N > MAXN)
        {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    }
    else
    {
        printf("Usage: %s <matrix_dimension> [random seed]\n", argv[0]);
        exit(0);
    }
}

/* Print matrix */
void print_matrix(float *matrix)
{
    int row, col;
    if (N <= 10)
    {
        printf("\n");
        for (row = 0; row < N; row++)
        {
            for (col = 0; col < N; col++)
            {
                printf("%8.2f\t", matrix[row * N + col]);
            }
            printf("\n");
        }
    }
}

/* Initialize host A and B */
void initialize_inputs()
{
    int row, col;

    printf("\nInitializing...\n");

    // Allocate memory for A and B on host device
    h_A = (float *)malloc(N*N * sizeof(float));
    h_B = (float *)malloc(N*N * sizeof(float));

    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            h_A[row * N + col] = (float)rand() / 32768.0;
            h_B[row * N + col] = 0.0;

        }
    }
}

#pragma endregion

int main (int argc, char **argv)
{
    #pragma region // host infrastructure

    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;

    /* Parameters and initialize inputs */
    parameters(argc, argv);
    initialize_inputs();

    /* Start the clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d\n", N);
    printf("\nStarting clock.\n");
    gettimeofday(&start, &tzdummy);

    #pragma endregion

    /* Print matrix N <= 10 */
    print_matrix(h_A);

    #pragma region // host infrastructure

    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
    
    
    /* Display timing results */
    printf("\nStopped clock.\n");
    printf("\nRuntime = %g ms.\n", (float)runtime/(float)1000);
    print_matrix(h_B);
    printf("\n---------------------------------------------\n");

    #pragma endregion
        
    free(h_A);
    free(h_B);
    exit(0);
}