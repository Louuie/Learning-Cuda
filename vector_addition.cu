#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Create the CUDA function
__global__ void addVectors(const float* a, const float* b, float* c, int n) {
    // Each GPU thread computes one element of the vector
    // This formula computes the global thread index
       // block index   // threads per block // thread index inside block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Here is where the actual computation happens
        // e.g. a = [1, 1, 1], b = [2, 2, 2] then the result is c = [3, 3, 3]
        c[i] = a[i] + b[i];
    }
}

// The main function, this is running on the CPU
int main() {
    int n = 1024; // 1024 elements, so we will add 1024 of them
    // Compute the memory size
    size_t bytes = n * sizeof(float);
    // Each float = 4 bytes
    // So the memory needed is:
        // 1024 * 4 = 4096 bytes
    // Host CPU Vectors
    std::vector<float> h_a(n, 1.0f); // 1024 elements of 1.0
    std::vector<float> h_b(n, 2.0f); // 1024 elements of 2.0
    std::vector<float> h_c(n, 0.0f); // this is the array where our result will be

    // Then we create float pointers for the gpu to reference, hence the pointer parameters in our AddVectors function
    float *d_a, *d_b, *d_c;
    // Allocate GPU Memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Now copy the data from the host (CPU) to the device (GPU) so it has the input array data
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int threads = 256; // each block will contain 256 threads
    int blocks = (n + threads - 1) / threads; // this ensures we launch enough threads
    // Launch the addVectors GPU Kernels
    addVectors<<<blocks, threads>>>(d_a, d_b, d_c, n);
    // So we launch 4 blocks, 256 threads, total threads = 1024. and each thread executes the addVectors

    // Now that we do the computation, we obviously can't access the vals of the device otherwise we would segfault, so we need to copy it over back to the host (CPU)
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Now that we have it on the CPU Memory, we can now print the result
    for (int i = 0; i < n; i++) {
        std::cout << "Iteration: " << i << " " << h_c[i] << std::endl;
    }
    // free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}