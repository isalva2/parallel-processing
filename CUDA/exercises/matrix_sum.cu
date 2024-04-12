#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

const size_t DSIZE = 16384; // matrix side dimension
const int block_size = 256; // CUDA maximum is 1024

// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds)
{
    /*
    the idx-th row of A corresponds to the idx-th thread
    given by the block index multiplied by the number of
    threads in a block, plus the thread index of the thread
    within the block
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ds)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < ds; i++)
        {
            /*
            Start at the idx-th row, first element, indexed at A[idx * ds],
            multiplying by ds moves down one row by a stride of ds.
            Next go one element over in the row ds times, to end of row.
            */
            sum += A[idx * ds + i];
        }
        sums[idx] = sum;
    }
}

// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds)
{
    /*
    the idx-th column of A corresponds to the idx-th thread
    given by the block index multiplied by the number of
    threads in a block, plus the thread index of the thread
    within the block
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ds)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < ds; i++)
        {
            /*
            Start at the idx-th column, first element, indexed at
            A[idx],then go down one row ds times, using a stride of
            ds (DSIZE). This stride is indexed using 1 * ds.
            */
            sum += A[idx + i * ds];
        }
        sums[idx] = sum;
    }
}

bool validate(float *data, size_t sz)
{
    for (size_t i = 0; i < sz; i++)
        if (data[i] != (float)sz)
        {
            printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz);
            return false;
        }
    return true;
}

int main()
{

    float *h_A, *h_sums, *d_A, *d_sums;
    h_A = new float[DSIZE * DSIZE]; // allocate space for data in host memory
    h_sums = new float[DSIZE]();

    for (int i = 0; i < DSIZE * DSIZE; i++) // initialize matrix in host memory
        h_A[i] = 1.0f;

    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float)); // allocate device space for A
    cudaMalloc(&d_sums, DSIZE* sizeof(float)); // allocate device space for vector d_sums
    cudaCheckErrors("cudaMalloc failure");       // error checking

    // copy matrix A to device:
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // cuda processing sequence step 1 is complete
    row_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    // cuda processing sequence step 2 is complete

    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    if (!validate(h_sums, DSIZE))
        return -1;
    printf("row sums correct!\n");

    cudaMemset(d_sums, 0, DSIZE * sizeof(float));

    column_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");
    // cuda processing sequence step 2 is complete

    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    if (!validate(h_sums, DSIZE))
        return -1;
    printf("column sums correct!\n");
    return 0;
}
