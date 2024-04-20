#include <stdio.h>
#include <assert.h>
#include <math.h>


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

#include <assert.h>
#include <math.h>

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
#pragma region // Global variables

#define N 512
complex A[N][N], B[N][N];
complex OUT[N][N];

#pragma endregion
#pragma region //IO

void read_data()
{
    int row, col;
    FILE *A_real, *B_real;
    A_real = fopen("data/2_im1", "r");
    B_real = fopen("data/2_im2", "r");
    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            fscanf(A_real, "%f", &A[row][col].r);
            fscanf(B_real, "%f", &B[row][col].r);
            // A[row][col].i = 0.0;
            // B[row][col].i = 0.0;
        }
    }
    fclose(A_real);
    fclose(B_real);
}

void write_output()
{
    int row, col;
    FILE *OUT_real;
    OUT_real = fopen("results/serial_results/serial_out_1", "w");
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

#pragma endregion

int main()
{
    int row, col;

    // Intermediate transpose
    complex A_T[N][N], B_T[N][N], OUT_temp;

    // Read in A and B from memory
    read_data();

    // Do fft on rows of A and B
    for (row = 0; row < N; row++)
    {
        c_fft1d(A[row], N, -1);
        c_fft1d(B[row], N, -1);

        // Transpose and store in new matrix
        for (col = 0; col < N; col++)
        {
            A_T[col][row] = A[row][col];
            B_T[col][row] = B[row][col];
        }
    }

    // Do fft on rows of A_T and B_T (columns of A and B)
    for (row = 0; row < N; row++)
    {
        c_fft1d(A_T[row], N, -1);
        c_fft1d(B_T[row], N, -1);
    }

    for (row = 0; row < N; row++)
    {
        // Perform point wise multiplication and transpose (correct orientation) to OUT_temp
        for (col = 0; col < N; col++)
        {
            // Try actual complex number multiplication
            OUT[col][row].r = A_T[row][col].r * B_T[row][col].r - A_T[row][col].i * B_T[row][col].i; 
            OUT[col][row].i = A_T[row][col].r * B_T[row][col].i + A_T[row][col].i * B_T[row][col].r;
        }
    }

    // IFFT on OUT and store transposed to OUT
    for (row = 0; row < N; row++)
    {
        c_fft1d(OUT[row], N, 1);
    }

    // Inplace transpose
    for (row = 1; row < N; row++)
    {
        for (col = 0; col < row; col++)
        {
            OUT_temp = OUT[row][col];
            OUT[row][col] = OUT[col][row];
            OUT[col][row] = OUT_temp;
        }
    }

    // Last IFFT and transpose store to OUT
    for (row = 0; row < N; row++)
    {
        c_fft1d(OUT[row], N, 1);
    }

    // Inplace transpose
    for (row = 1; row < N; row++)
    {
        for (col = 0; col < row; col++)
        {
            OUT_temp = OUT[row][col];
            OUT[row][col] = OUT[col][row];
            OUT[col][row] = OUT_temp;
        }
    }

    // Write out
    write_output();
    
    return 0;
}