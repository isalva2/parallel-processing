#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <mpi.h>

#define N 8
int numprocs, myid;


// Complex struct
typedef struct {float r; float i;} complex;
complex A[N][N], B[N][N], OUT[N][N];

#pragma region // Source FFT1D

/*
 ------------------------------------------------------------------------
 FFT1D            c_fft1d(r,i,-1)
 Inverse FFT1D    c_fft1d(r,i,+1)
 ------------------------------------------------------------------------
*/
/* ---------- FFT 1D
   This computes an in-place complex-to-complex FFT
   r is the real and imaginary arrays of n=2^m points.
   isign = -1 gives forward transform
   isign =  1 gives inverse transform
*/


static complex ctmp;

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

void c_fft1d(complex *r, int      n, int      isign)
{
   int     m,i,i1,j,k,i2,l,l1,l2;
   float   c1,c2,z;
   complex t, u;

   if (isign == 0) return;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j)
         C_SWAP(r[i], r[j]);
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* m = (int) log2((double)n); */
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;

            /* t = u * r[i1] */
            t.r = u.r * r[i1].r - u.i * r[i1].i;
            t.i = u.r * r[i1].i + u.i * r[i1].r;

            /* r[i1] = r[i] - t */
            r[i1].r = r[i].r - t.r;
            r[i1].i = r[i].i - t.i;

            /* r[i] = r[i] + t */
            r[i].r += t.r;
            r[i].i += t.i;
         }
         z =  u.r * c1 - u.i * c2;

         u.i = u.r * c2 + u.i * c1;
         u.r = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (isign == -1) /* FWD FFT */
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for inverse transform */
   if (isign == 1) {       /* IFFT*/
      for (i=0;i<n;i++) {
         r[i].r /= n;
         r[i].i /= n;
      }
   }
}


#pragma endregion
#pragma region //IO

void read_data()
{
    int row, col;
    FILE *A_real, *B_real;
    A_real = fopen(".test_data/1_im1", "r");
    B_real = fopen(".test_data/1_im2", "r");
    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            fscanf(A_real, "%f", &A[row][col].r);
            fscanf(B_real, "%f", &B[row][col].r);
            A[row][col].i = 0.0;
            B[row][col].i = 0.0;
        }
    }
    fclose(A_real);
    fclose(B_real);
}

void write_output()
{
    int row, col;
    FILE *OUT_real;
    OUT_real = fopen("results/spmd_1/serial_out_1", "w");
    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            fprintf(OUT_real, "%e\t", OUT[row][col].r);
        }
        fprintf(OUT_real, "\n");
    }
    fclose(OUT_real);
}

// Testing
void write_A()
{
    int row, col;
    FILE *OUT_real;
    OUT_real = fopen("results/spmd_1/serial_out_1", "w");
    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            fprintf(OUT_real, "%e\t", A[row][col].r);
        }
        fprintf(OUT_real, "\n");
    }
    fclose(OUT_real);
}

#pragma endregion
#pragma region // Debug

void print_A()
{
    printf("Matrix:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%8.2f;   ", A[i][j].r);
        }
        printf("\n");
    }
}

#pragma endregion

int main(int argc, char *argv[])
{
    // Initialize MPI environment
    MPI_Init(&argc , &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // MPI variables
    int row, col, proc, worker_proc;
    MPI_Status status;

    // Worker node variables
    int block_size = (int) N / (numprocs);
    complex worker_buffer[block_size * N];

    #pragma region // Custom datatype declarations

    /* 
    Create MPI datatype for complex numbers
    */
    MPI_Datatype complex_type;
    int lengths[2] = {1, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(complex, r);
    offsets[1] = offsetof(complex, i);
    MPI_Datatype types[2] = {MPI_FLOAT, MPI_FLOAT};
    MPI_Type_create_struct(2, lengths, offsets, types, &complex_type);
    MPI_Type_commit(&complex_type);    

    /* 
    Create MPI datatype for row vectors of complex numbers
    */
    MPI_Datatype row_type;
    MPI_Type_vector(block_size, block_size * N, 1, complex_type, &row_type);
    MPI_Type_commit(&row_type);

    /*
    Create MPI datatype for column vectors of complex numbers
    */
    MPI_Datatype column_type;
    MPI_Type_vector(N, block_size, N, complex_type, &column_type);
    MPI_Type_commit(&column_type);

    /*
    Flattened 1-D vector of complex_type
    */
    MPI_Datatype flat_column;
    MPI_Type_contiguous(block_size * N, complex_type, &flat_column);
    MPI_Type_commit(&flat_column);

    #pragma endregion


    // Root IO
    if (myid == 0)
    {
        printf("Matrix A before calcs:\n\n");
        read_data();
        print_A();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("barrier\n");
    printf("\n");


    
    /*
    This communicate-calculate block:
    1) Sends columns of A (1-D static block scheduling)
    2) Performs column-wise calcs on A
    3) Sends back columns of A TRANSPOSED

    i.e. A -> calcs -> A_t
    */
    int lower_bound;
    if (myid == 0)
    {
        // Send blocked columns of A to each proc
        for (proc = 1; proc < numprocs; proc++)
        {
            //1-D static block scheduling
            lower_bound = (proc) * N / (numprocs);
            MPI_Send(&A[0][lower_bound], 1, column_type, proc, 2, MPI_COMM_WORLD);
        }

        // Naive transpose in-place the first block_size-th columns of root's copy of A
        complex a_temp;
        for (row = 0; row < block_size; row ++)
        {
            for (col = 0; col < row; col++)
            {
                a_temp = A[row][col];
                A[row][col] = A[col][row];
                A[col][row] = a_temp;
            }
        }
        // Calc using 1D-fft

        // Receive worked on, transposed COLUMNS of A as ROWS (make A_t)
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Recv(&A[block_size * proc], 1, row_type, proc, 3, MPI_COMM_WORLD, &status);
        }
    }
    else
    {
        // Receive columns of A from root as a flattened vector of complex
        MPI_Recv(&worker_buffer, 1, flat_column, 0, 2, MPI_COMM_WORLD, &status);
        
        // Move complexes in 1-D buffer to workers' copies of A
        int buf_idx;
        for (buf_idx = 0; buf_idx < block_size * N; buf_idx++)
        {
            A[buf_idx % block_size + block_size * myid][(int)floor(buf_idx / block_size)] = worker_buffer[buf_idx];
        }
        // Worker calc using 1D-fft

        // Send worked on, transposed, COLUMNS of A back as ROWS
        MPI_Send(&A[block_size * myid], 1, row_type, 0, 3, MPI_COMM_WORLD);
    }

    // This communication-calculation block sends along rows 
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (myid == 0)
    // {
    //     for (proc = 1; proc < numprocs; proc++)
    //     {
    //         lower_bound = (proc - 1) * N / (numprocs - 1);
    //         MPI_Recv(&A[lower_bound * N], 1, row_type, proc, 1, MPI_COMM_WORLD, &status);
    //     }
    // }
    // else
    // {
    //     lower_bound = (myid - 1) * N / (numprocs - 1);
    //     MPI_Send(&A[lower_bound * N][0], 1, row_type, 0, 1, MPI_COMM_WORLD);
    // }

    if (myid == 0)
    {
        write_A();
        print_A();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 1; i < numprocs; i++)
    {
        if (myid == i)
        {
            printf("\n");
            printf("My ID = %d\n\n", myid);
            print_A();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
    {
        printf("\nMatrix A after Calc:\n");
        print_A();
    }

    MPI_Type_free(&complex_type);
    MPI_Type_free(&row_type);
    MPI_Type_free(&column_type);
    MPI_Type_free(&flat_column);
    MPI_Finalize();
    exit(0);
}