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

typedef struct
{
    float r;
    float i;
} complex;
static complex ctmp;

#define C_SWAP(a, b) \
    {                \
        ctmp = (a);  \
        (a) = (b);   \
        (b) = ctmp;  \
    }

void c_fft1d(complex *r, int n, int isign)
{
    int m, i, i1, j, k, i2, l, l1, l2;
    float c1, c2, z;
    complex t, u;

    if (isign == 0)
        return;

    /* Do the bit reversal */
    i2 = n >> 1;
    j = 0;
    for (i = 0; i < n - 1; i++)
    {
        if (i < j)
            C_SWAP(r[i], r[j]);
        k = i2;
        while (k <= j)
        {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* m = (int) log2((double)n); */
    for (i = n, m = 0; i > 1; m++, i /= 2)
        ;

    /* Compute the FFT */
    c1 = -1.0;
    c2 = 0.0;
    l2 = 1;
    for (l = 0; l < m; l++)
    {
        l1 = l2;
        l2 <<= 1;
        u.r = 1.0;
        u.i = 0.0;
        for (j = 0; j < l1; j++)
        {
            for (i = j; i < n; i += l2)
            {
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
            z = u.r * c1 - u.i * c2;

            u.i = u.r * c2 + u.i * c1;
            u.r = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (isign == -1) /* FWD FFT */
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    /* Scaling for inverse transform */
    if (isign == 1)
    { /* IFFT*/
        for (i = 0; i < n; i++)
        {
            r[i].r /= n;
            r[i].i /= n;
        }
    }
}

#pragma endregion

#define N 512
complex A[N][N], B[N][N];

void read_data()
{
    int row, col;
    FILE *A_real, *A_imag, *B_real, *B_imag;


    A_real = fopen("data/1_im1", "r");
    A_imag = fopen("data/2_im1", "r");
    B_real = fopen("data/1_im2", "r");
    B_imag = fopen("data/2_im2", "r");

    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            fscanf(A_real, "%f", &A[row][col].r);
            fscanf(A_imag, "%f", &A[row][col].i);
            fscanf(B_real, "%f", &B[row][col].r);
            fscanf(B_imag, "%f", &B[row][col].i);
        }
    }
}

int  main()
{
    read_data();
    for (int i = 0; i < 10; i ++)
    {
        printf("%f\t", A[0][i].i);
    }
    return 0;
}