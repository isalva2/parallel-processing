#include <stdio.h>

__global__ void hello()
{

    printf("Hello from block: %d, thread: %d\n", blockIdx.x, threadIdx.x);
    printf("test");
}

int main()
{

    hello<<<2, 2>>>();
    cudaDeviceSynchronize();
    return(0);
}
