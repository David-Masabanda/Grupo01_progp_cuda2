#include <vector>
#include <chrono>
#include <fmt/core.h>
#include <cuda_runtime.h>

extern "C" void cuda_scanPrefixSum(const int* input, int* output, int n);

int main() {
    // Ejemplo de uso
    std::vector<int> input = {10, 20, 10, 5, 15,12,118,59,64,75,216,84,262,84,4,15,7};
    std::vector<int> output(input.size());

    auto start = std::chrono::high_resolution_clock::now();

    // Copiar datos al dispositivo
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, input.size() * sizeof(int));
    cudaMalloc(&d_output, input.size() * sizeof(int));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Llamar a la función de Scan
    cuda_scanPrefixSum(d_input, d_output, input.size());

    // Copiar resultados de vuelta al host
    cudaMemcpy(output.data(), d_output, input.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria del dispositivo
    cudaFree(d_input);
    cudaFree(d_output);

    // Imprimir el resultado
    fmt::print("Input: ");
    for (const auto& value : input) {
        fmt::print("{} ", value);
    }
    fmt::print("\n");

    fmt::print("Prefix Sum: ");
    for (const auto& value : output) {
        fmt::print("{} ", value);
    }
    fmt::print("\n");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiempo = end - start;
    fmt::print("Tiempo de proceso: {}\n", tiempo.count());

    return 0;
}