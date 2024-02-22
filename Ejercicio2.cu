#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <fmt/core.h>

namespace ch = std::chrono;

std::vector<int> read_file() {
    std::fstream fs("C:/Users/josue/Desktop/Universidad/ProgParalela/Grupal_final/datos/datos4.txt", std::ios::in);
    std::string line;
    std::vector<int> ret;
    while (std::getline(fs, line)) {
        ret.push_back(std::stoi(line));
    }
    fs.close();
    return ret;
}

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

void cuda_scanPrefixSum(const int* input, int* output, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(int);

    scanPrefixSumKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, output, n);
}

int main() {
    // Ejemplo de uso
    std::vector<int> input = {10, 20, 10, 5, 15,12,118,59,64,75,216,84,262,84,4,15,7};
    //std::vector<int> input = read_file();
    std::vector<int> output(input.size());

    // Copiar datos al dispositivo
    auto start = std::chrono::high_resolution_clock::now();

    int *d_input;
    int *d_output;
    cudaMalloc(&d_input, input.size() * sizeof(int));
    cudaMalloc(&d_output, input.size() * sizeof(int));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiempo = end - start;
    fmt::print("Tiempo de copia al dispositivo: {}\n", tiempo.count());

    // Llamar a la función de Scan
    start = std::chrono::high_resolution_clock::now();

    cuda_scanPrefixSum(d_input, d_output, input.size());
    end = std::chrono::high_resolution_clock::now();
    tiempo = end - start;
    fmt::print("Tiempo de llamada a la funcion de Scan: {}\n", tiempo.count());

    // Copiar resultados de vuelta al host
    start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(output.data(), d_output, input.size() * sizeof(int), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    tiempo = end - start;
    fmt::print("Tiempo resultados de vuelta al host: {}\n", tiempo.count());

    // Liberar memoria del dispositivo
    cudaFree(d_input);
    cudaFree(d_output);

    // Imprimir el resultado
//    fmt::print("Input: ");
//    for (const auto& value : input) {
//        fmt::print("{} ", value);
//    }
//    fmt::print("\n");
//
//    fmt::print("Prefix Sum: ");
//    for (const auto& value : output) {
//        fmt::print("{} ", value);
//    }
//    fmt::print("\n");

    return 0;
}