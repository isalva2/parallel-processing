#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <mpi.h>

#define N 512

#pragma region // MARK: source 1D-fft

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

typedef struct {float r; float i;} complex;
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
#pragma region // MARK: IO

void read_file(char path[20], complex matrix[N][N])
{
    FILE *fp = fopen(path, "r");
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            fscanf(fp, "%f", &matrix[row][col].r);
            matrix[row][col].i = 0.0;
        }
    }
    fclose(fp);
}

void write_file(char path[], complex matrix[N][N])
{
    int row, col;
    FILE *fp = fopen(path, "w");
    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            fprintf(fp, "%e\t", matrix[row][col].r);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

#pragma endregion
#pragma region // MARK: Helper functions

void transpose(complex a[N][N])
{
    int row, col;
    for (row = 0; row < N; row++)
    {
        for (col = row + 1; col < N; col++)
        { 
            complex temp = a[row][col];
            a[row][col] = a[col][row];
            a[col][row] = temp;
        }
    }
}

void hadamard_product(complex a[N][N], complex b[N][N], complex out[N][N], int lower_bound, int upper_bound)
{
    for (int row = lower_bound; row < upper_bound; row++)
    {
        for (int col = 0; col < N; col++)
        {
            out[row][col].r = a[row][col].r * b[row][col].r - a[row][col].i * b[row][col].i; 
            out[row][col].i = a[row][col].r * b[row][col].i + a[row][col].i * b[row][col].r;
        }
    }
}

#pragma endregion
#pragma region // MARK: Debug

void print_matrix(complex matrix[N][N], int id)
{
    printf("Matrix: %d\n", id);
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            printf("%6.2f;\t", matrix[row][col].r);
        }
        printf("\n");
    }
}

#pragma endregion

int main(int argc, char *argv[])
{
    // File paths
    // char image_1[] = "data/test_data/1_im1";
    // char image_2[] = "data/test_data/1_im2";
    // char output[] = "output";

    char image_1[] = "data/2_im1";
    char image_2[] = "data/2_im2";
    // char output[] = "data/results/output";

    // MPI Variables
    int myid, numprocs;

    // Initialize MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Scheduling
    int row, col, proc;
    int block_size = (int) N/numprocs;
    int lower_bound = block_size * myid;
    int upper_bound = lower_bound + block_size;

    // Initialize variables
    complex A[N][N], B[N][N], OUT[N][N];

    // Performance logging
    double start_t, end_t;
    double start_calc1, stop_calc1; // First 1D-fft
    double start_calc2, stop_calc2; // Transpose
    double start_calc3, stop_calc3; // Second 1D-fft
    double start_calc4, stop_calc4; // Hadamard product
    double start_calc5, stop_calc5; // First inverse 1D-fft
    double start_calc6, stop_calc6; // Transpose
    double start_calc7, stop_calc7; // Second inverse 1D-fft
    double start_last_calc;         // Last calc


    // Custom complex MPI datatype
    MPI_Datatype complex_type;
    int lengths[2] = {1, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(complex, r);
    offsets[1] = offsetof(complex, i);
    MPI_Datatype types[2] = {MPI_FLOAT, MPI_FLOAT};
    MPI_Type_create_struct(2, lengths, offsets, types, &complex_type);
    MPI_Type_commit(&complex_type);

    // Custom vector MPI datatype
    MPI_Datatype row_type;
    MPI_Type_vector(block_size, N, N, complex_type, &row_type);
    MPI_Type_commit(&row_type);

    // IO Printout
    if (myid == 0)
    {
        printf("2D Convolution\n");
        printf("Computing using MPI with %d processes\n", numprocs);
        printf("Starting...\n\n");
    }

    // Send/recv tags
    int tag1= 1, tag2 = 2;
    MPI_Status status;

    // MARK: First 1D-fft on A and B on rows
    if (myid == 0)
    {
        // root gets data
        read_file(image_1, A);
        read_file(image_2, B);

        // Start clock after input
        start_t = MPI_Wtime();

        // Send data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Send(&A[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD);
            MPI_Send(&B[block_size * proc], 1, row_type, proc, tag2, MPI_COMM_WORLD);
        }

        // Record first calc
        start_calc1 = MPI_Wtime();
    }
    else
    {
        MPI_Recv(&A[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[block_size * myid], 1, row_type, 0, tag2, MPI_COMM_WORLD, &status);
        
    }

    for (row = lower_bound; row < upper_bound; row++)
    {
        c_fft1d(A[row], N, -1);
        c_fft1d(B[row], N, -1);
    }

    if (myid == 0)
    {
        // Record first calc
        stop_calc1 = MPI_Wtime();

        // Recv data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Recv(&A[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD, &status);
            MPI_Recv(&B[block_size * proc], 1, row_type, proc, tag2, MPI_COMM_WORLD, &status);
        }
    }
    else
    {
        MPI_Send(&A[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD);
        MPI_Send(&B[block_size * myid], 1, row_type, 0, tag2, MPI_COMM_WORLD);
    }

    // MARK: Second 1D-fft on transposed A and B (Columns of A and B)
    if (myid == 0)
    {
        // Record calc
        start_calc2 = MPI_Wtime();

        transpose(A);
        transpose(B);

        // Record calc
        stop_calc2 = MPI_Wtime();

        // Send data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Send(&A[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD);
            MPI_Send(&B[block_size * proc], 1, row_type, proc, tag2, MPI_COMM_WORLD);
        }

        // Record calc
        start_calc3 = MPI_Wtime();
    }
    else
    {
        MPI_Recv(&A[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[block_size * myid], 1, row_type, 0, tag2, MPI_COMM_WORLD, &status);
    }

    for (row = lower_bound; row < upper_bound; row++)
    {
        c_fft1d(A[row], N, -1);
        c_fft1d(B[row], N, -1);
    }

    if (myid == 0)
    {
        // Record calc
        stop_calc3 = MPI_Wtime();

        // Recv data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Recv(&A[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD, &status);
            MPI_Recv(&B[block_size * proc], 1, row_type, proc, tag2, MPI_COMM_WORLD, &status);
        }

    }
    else
    {
        MPI_Send(&A[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD);
        MPI_Send(&B[block_size * myid], 1, row_type, 0, tag2, MPI_COMM_WORLD);
    }

    // Mark: Hadamard product on root and transpose from OUT_T to OUT
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
    {
        // Record calc
        start_calc4 = MPI_Wtime();

        hadamard_product(A, B, OUT, 0, N);
        transpose(OUT);
        
        // Record calc
        stop_calc4 = MPI_Wtime();
    }

    // MARK: Inverse 1D-fft on rows of OUT
    if (myid == 0)
    {
        // Send data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Send(&OUT[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD);
        }

        // Record calc
        start_calc5 = MPI_Wtime();
    }
    else
    {
        MPI_Recv(&OUT[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD, &status);
        
    }

    for (row = lower_bound; row < upper_bound; row++)
    {
        c_fft1d(OUT[row], N, 1);
    }

    if (myid == 0)
    {
        // Record calc
        stop_calc5 = MPI_Wtime();
                
        // Recv data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Recv(&OUT[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD, &status);
        }
    }
    else
    {
        MPI_Send(&OUT[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD);
    }

    // MARK: Second 1D-fft on transposed A and B (Columns of A and B)
    if (myid == 0)
    {
        // Record calc
        start_calc6 = MPI_Wtime();

        transpose(OUT);

        // Record calc
        stop_calc6 = MPI_Wtime();

        // Send data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Send(&OUT[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD);
        }

        // Record calc
        start_calc7 = MPI_Wtime();
    }
    else
    {
        MPI_Recv(&OUT[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD, &status);
    }

    for (row = lower_bound; row < upper_bound; row++)
    {
        c_fft1d(OUT[row], N, 1);
    }

    if (myid == 0)
    {
        // Record calc
        stop_calc7 = MPI_Wtime();

        // Recv data
        for (proc = 1; proc < numprocs; proc++)
        {
            MPI_Recv(&OUT[block_size * proc], 1, row_type, proc, tag1, MPI_COMM_WORLD, &status);
        }

    }
    else
    {
        MPI_Send(&OUT[block_size * myid], 1, row_type, 0, tag1, MPI_COMM_WORLD);
    }

    // MARK: IO Out
    if (myid == 0)
    {
        // Record calc
        start_last_calc = MPI_Wtime();

        // Transpose OUT
        transpose(OUT);

        // Record end of algo
        end_t = MPI_Wtime();

        // Write out
        write_file("test_output", OUT);

        // Print out times
        double calc1 = stop_calc1 - start_calc1;
        double calc2 = stop_calc2 - start_calc2;
        double calc3 = stop_calc3 - start_calc3;
        double calc4 = stop_calc4 - start_calc4;
        double calc5 = stop_calc5 - start_calc5;
        double calc6 = stop_calc6 - start_calc6;
        double calc7 = stop_calc7 - start_calc7;
        double last_calc = end_t - start_last_calc;

        double total_runtime = end_t - start_t;
        double total_calc_time = calc1 + calc2 + calc3 + calc4 + calc5 + calc6 + calc7 + last_calc;
        double total_comm_time = total_runtime - total_calc_time;

        printf("Calc time:\t%f sec\n", total_calc_time);
        printf("Comm time:\t%f sec\n", total_comm_time);
        printf("Runtime:\t%f sec\n", total_runtime);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&row_type);
    MPI_Type_free(&complex_type);
    MPI_Finalize();
    exit(0);
}