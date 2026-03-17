cmake_minimum_required(VERSION 3.28)
project(CUDA_Development)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 121)
set(CMAKE_CUDA_FLAGS "-arch=sm_121")

enable_language(CUDA)

add_executable(CUDA_Development main.cu)

set_target_properties(CUDA_Development PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)