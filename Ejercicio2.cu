#include <cuda_runtime.h>

__global__ void scanPrefixSumKernel(const int* input, int* output, int n) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Copiar datos a la memoria compartida
    temp[tid] = (index < n) ? input[index] : 0;
    __syncthreads();

    // Realizar la suma acumulativa
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();

        temp[tid] += val;
        __syncthreads();
    }

    // Escribir el resultado de vuelta en la memoria global
    if (index < n) {
        output[index] = temp[tid];
    }
}

extern "C" void cuda_scanPrefixSum(const int* input, int* output, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(int);

    scanPrefixSumKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, output, n);
}