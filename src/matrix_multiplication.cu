#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void matrixMultiplication(const float* matrix_A, const float* matrix_B, float* matrix_C,
                                     int rowsA, int colsA, int colsB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column in C
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row in C

    if (y < rowsA && x < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += matrix_A[y * colsA + k] * matrix_B[k * colsB + x];
        }
        matrix_C[y * colsB + x] = sum;
    }
}

// int main() {
//     int rowsA = 2;
//     int colsA = 3;
//
//     int rowsB = 3;
//     int colsB = 2;
//
//     size_t bytesA = rowsA * colsA * sizeof(float);
//     size_t bytesB = rowsB * colsB * sizeof(float);
//     size_t bytesC = rowsA * colsB * sizeof(float);
//
//     std::vector<float> h_A = {
//         1, 2, 3,
//         4, 5, 6
//     }; // 2x3
//
//     std::vector<float> h_B = {
//         6, 5,
//         4, 3,
//         2, 1
//     }; // 3x2
//
//     std::vector<float> h_C(rowsA * colsB, 0.0f); // 2x2
//
//     float *d_a, *d_b, *d_c;
//
//     cudaMalloc(&d_a, bytesA);
//     cudaMalloc(&d_b, bytesB);
//     cudaMalloc(&d_c, bytesC);
//
//     cudaMemcpy(d_a, h_A.data(), bytesA, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_B.data(), bytesB, cudaMemcpyHostToDevice);
//
//     dim3 threads(16, 16);
//     dim3 blocks((colsB + threads.x - 1) / threads.x,
//                 (rowsA + threads.y - 1) / threads.y);
//
//     matrixMultiplication<<<blocks, threads>>>(d_a, d_b, d_c, rowsA, colsA, colsB);
//
//     cudaMemcpy(h_C.data(), d_c, bytesC, cudaMemcpyDeviceToHost);
//
//     for (int row = 0; row < rowsA; row++) {
//         for (int col = 0; col < colsB; col++) {
//             std::cout << h_C[row * colsB + col] << " ";
//         }
//         std::cout << "\n";
//     }
//
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
//
//     return 0;
// }