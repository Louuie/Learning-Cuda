#pragma once

__global__ void matrixMultiplication(const float* A, const float* B, float* C,
                                     int rowsA, int colsA, int colsB);