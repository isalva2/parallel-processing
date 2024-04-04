#include <stdio.h>

// Single-precision A*X plus Y
// From nvidia's website: https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y, *x_d, *y_d;
    x = (float*) malloc(N*sizeof(float));
    y = (float*) malloc(N*sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements (lol)
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, x_d, y_d);

    cudaMemcpy(y, y_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i; i < N; i++)
    {
        maxError = max(maxError, abs(y[i] - 4.0f));
    }
    printf("Max error: %f\n", maxError);

    free(x); free(y);
    cudaFree(x_d); cudaFree(y_d);
}