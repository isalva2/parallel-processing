#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <mpi.h>

#define N 512

// MARK: source 1D-fft

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

// MARK: IO

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

// MARK: Helper fns

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

// MARK: main
int main(int argc, char *argv[])
{
    // File paths
    // char image_1[] = "data/test_data/1_im1";
    // char image_2[] = "data/test_data/1_im2";
    char image_1[] = "data/2_im1";
    char image_2[] = "data/2_im2";
    char output[] = "data/results/collective_call";

    // MPI Variables
    int myid, numprocs;

    // Initialize MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // MARK: New Communicator

    // Early exit if numprocs != 8
    if (numprocs != 8)
    {
        if (myid == 0)
            printf("Task and Data Parallel Model needs 8 processes :(\n");
        MPI_Finalize();
        exit(1);
    }

    // Get color and key for Comm_split
    int group_size = 2;
    int color = myid / group_size;
    int key = myid % group_size;

    // Split into duos
    MPI_Comm NEW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &NEW_COMM);

    // Get new rank in communicator
    int newid;
    MPI_Comm_rank(NEW_COMM, &newid);

    // Scheduling
    int row, col, proc;
    int block_size = (int) N / group_size;
    int block_count = block_size * N;
    int lower_bound = newid * N;
    int upper_bound = lower_bound + block_size;

    // Initialize variables
    complex A[N][N], B[N][N], OUT[N][N];

    // MARK: Timing 
    double start_t, end_t;
    double start_calc1, stop_calc1; // First 1D-fft
    double start_calc2, stop_calc2; // Transpose
    double start_calc3, stop_calc3; // Second 1D-fft
    double start_calc4, stop_calc4; // Hadamard product
    double start_calc5, stop_calc5; // First inverse 1D-fft
    double start_calc6, stop_calc6; // Transpose
    double start_calc7, stop_calc7; // Second inverse 1D-fft
    double start_last_calc;         // Last calc

    // MARK: MPI Datatypes
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

    // Send/Recv tags
    int tag = 1, tag2 = 2;
    MPI_Status status;

    // testing
    printf("Old id: %d, New id: %d, My group: %d\n", myid, newid, color);

    MPI_Barrier(NEW_COMM);

    // MARK: Start
    if (myid == 0)
    {
        printf("\n2D Convolution\n");
        printf("Computing using task and data parallel model with %d processes\n", numprocs);
        printf("MPI Communicator with %d groups and %d processes per group\n", numprocs / group_size, group_size);
        printf("Starting...\n\n");
    }

    // MARK: Group 1
    // 2D-fft on A
    if (color == 0)
    {
        if (newid == 0)
        {
            read_file(image_1, A);
            MPI_Send(A[block_size], 1, row_type, 1, tag, NEW_COMM);
        }
        else
        {
            MPI_Recv(A[0], 1, row_type, 0, tag, NEW_COMM, &status);
        }

        for (row = 0; row < block_size; row++)
        {
            c_fft1d(A[row], N, -1);
        }

        if (newid == 0)
        {
            MPI_Recv(A[block_size], 1, row_type, 1, tag, NEW_COMM, &status);
            transpose(A);
            MPI_Send(A[block_size], 1, row_type, 1, tag, NEW_COMM);

        }
        else
        {
            MPI_Send(A[0], 1, row_type, 0, tag, NEW_COMM);
            MPI_Recv(A[0], 1, row_type, 0, tag, NEW_COMM, &status);
        }

        for (row = 0; row < block_size; row++)
        {
            c_fft1d(A[row], N, -1);
        }

        if (newid == 0)
        {
            MPI_Recv(A[block_size], 1, row_type, 1, tag, NEW_COMM, &status);
        }
        else
        {
            MPI_Send(A[0], 1, row_type, 0, tag, NEW_COMM);
        }
    }

    // MARK: Group 2
    // 2D-fft on B
    if (color == 1)
    {
        if (newid == 0)
        {
            read_file(image_2, B);
            MPI_Send(B[block_size], 1, row_type, 1, tag, NEW_COMM);
        }
        else
        {
            MPI_Recv(B[0], 1, row_type, 0, tag, NEW_COMM, &status);
        }

        for (row = 0; row < block_size; row++)
        {
            c_fft1d(B[row], N, -1);
        }

        if (newid == 0)
        {
            MPI_Recv(B[block_size], 1, row_type, 1, tag, NEW_COMM, &status);
            transpose(B);
            MPI_Send(B[block_size], 1, row_type, 1, tag, NEW_COMM);

        }
        else
        {
            MPI_Send(B[0], 1, row_type, 0, tag, NEW_COMM);
            MPI_Recv(B[0], 1, row_type, 0, tag, NEW_COMM, &status);
        }

        for (row = 0; row < block_size; row++)
        {
            c_fft1d(B[row], N, -1);
        }

        if (newid == 0)
        {
            MPI_Recv(B[block_size], 1, row_type, 1, tag, NEW_COMM, &status);
        }
        else
        {
            MPI_Send(B[0], 1, row_type, 0, tag, NEW_COMM);
        }
    }

    // MARK: Global Comm
    /*
    We switch back to using MPI_COMM_WORLD for intergroup communication.
    
    In this case we want Group 1 and Group 2 to send there copies of A and B
    respectively to Group 3. The rank 0 processes will communicate with each
    other using there old global ranks. I.e., old ranks 0 and 2 send data to old rank 4.
    
    */
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0)
        MPI_Send(A, N * N, complex_type, 4, tag, MPI_COMM_WORLD);
    if (myid == 2)
        MPI_Send(B, N * N, complex_type, 4, tag, MPI_COMM_WORLD);
    if (myid == 4)
    {
        MPI_Recv(A, N * N, complex_type, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(B, N * N, complex_type, 2, tag, MPI_COMM_WORLD, &status);
    }

    // MARK: Group 3
    if (color == 2)
    {
        if (newid == 0)
        {
            MPI_Send(A[block_size], 1, row_type, 1, tag,  NEW_COMM);
            MPI_Send(B[block_size], 1, row_type, 1, tag2, NEW_COMM);
        }
        else
        {
            MPI_Recv(A[0], 1, row_type, 0, tag,  NEW_COMM, &status);
            MPI_Recv(B[0], 1, row_type, 0, tag2, NEW_COMM, &status);
        }

        hadamard_product(A, B, OUT, 0, block_size);

        if (newid == 0)
        {
            MPI_Recv(OUT[block_size], 1, row_type, 1, tag, NEW_COMM, &status);
            transpose(OUT);
        }
        else
        {
            MPI_Send(OUT[0], 1, row_type, 0, tag, NEW_COMM);
        }
    }

    // MARK: Global Comm
    if (myid == 4)
        MPI_Send(OUT[0], N * N, complex_type, 6, tag, MPI_COMM_WORLD);
    if (myid == 6)
        MPI_Recv(OUT[0], N * N, complex_type, 4, tag, MPI_COMM_WORLD, &status);

    // MARK: Group 4
    // inverse 2D-fft on OUT
    if (color == 3)
    {
        if (newid == 0)
        {
            MPI_Send(OUT[block_size], 1, row_type, 1, tag, NEW_COMM);
        }
        else
        {
            MPI_Recv(OUT[0], 1, row_type, 0, tag, NEW_COMM, &status);
        }

        for (row = 0; row < block_size; row++)
        {
            c_fft1d(OUT[row], N, 1);
        }

        if (newid == 0)
        {
            MPI_Recv(OUT[block_size], 1, row_type, 1, tag, NEW_COMM, &status);
            transpose(OUT);
            MPI_Send(OUT[block_size], 1, row_type, 1, tag, NEW_COMM);

        }
        else
        {
            MPI_Send(OUT[0], 1, row_type, 0, tag, NEW_COMM);
            MPI_Recv(OUT[0], 1, row_type, 0, tag, NEW_COMM, &status);
        }

        for (row = 0; row < block_size; row++)
        {
            c_fft1d(OUT[row], N, 1);
        }

        if (newid == 0)
        {
            MPI_Recv(OUT[block_size], 1, row_type, 1, tag, NEW_COMM, &status);
            transpose(OUT);
            write_file("Test_OUT", OUT);
        }
        else
        {
            MPI_Send(OUT[0], 1, row_type, 0, tag, NEW_COMM);
        }
    }



    // MARK: IO Out
    // if (myid == 0)
    // {
    //     // Record calc
    //     start_last_calc = MPI_Wtime();

    //     // Transpose OUT
    //     transpose(OUT);

    //     // Record end of algo
    //     end_t = MPI_Wtime();

    //     // Write out
    //     write_file(output, OUT);

    //     // Print out times
    //     double calc1 = stop_calc1 - start_calc1;
    //     double calc2 = stop_calc2 - start_calc2;
    //     double calc3 = stop_calc3 - start_calc3;
    //     double calc4 = stop_calc4 - start_calc4;
    //     double calc5 = stop_calc5 - start_calc5;
    //     double calc6 = stop_calc6 - start_calc6;
    //     double calc7 = stop_calc7 - start_calc7;
    //     double last_calc = end_t - start_last_calc;

    //     double total_runtime = end_t - start_t;
    //     double total_calc_time = calc1 + calc2 + calc3 + calc4 + calc5 + calc6 + calc7 + last_calc;
    //     double total_comm_time = total_runtime - total_calc_time;

    //     printf("Calc time:\t%f sec\n", total_calc_time);
    //     printf("Comm time:\t%f sec\n", total_comm_time);
    //     printf("Runtime:\t%f sec\n", total_runtime);
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&row_type);
    MPI_Type_free(&complex_type);
    MPI_Finalize();
    exit(0);
}