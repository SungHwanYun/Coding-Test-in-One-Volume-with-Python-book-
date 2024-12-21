#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<bits/stdc++.h>
using namespace std;

#define CHECK(call) \
{ \
    const cudaError_t error = call;\
    if (error != cudaSuccess) { \
        printf("[device] Error: %s %d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    } \
}

// Check if the computation results of the CPU and GPU are the same
void checkResultInt(int* h_data, int *d_data, const int n) {
    int match = 1;
    for (int i = 0; i < n; i++) {
        if (h_data[i] != d_data[i]) {
            match = 0;
            printf("[host] Arrays do not match!\n");
            printf("[host] host %5d gpu %5d at current %d\n", h_data[i], d_data[i], i);
            break;
        }
    }
    if (match) printf("[host] Arrays match.\n\n");
}

__global__ void sumArrayElementK2D1D(int* g_idata, int* g_odata, unsigned int nx, unsigned int ny, int k) {
    unsigned int tid = threadIdx.x;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // blockDim.y(=1), threadIdx.y(=0)
    unsigned int iy = blockIdx.y; // blockDim.y(=1), threadIdx.y(=0)
    unsigned int idx = iy * nx + ix;

    // boundary check
    if (ix >= nx || iy >= ny) return;

    // branch divergence!!!
    if (g_idata[idx] == k) g_idata[idx] = 1;
    else g_idata[idx] = 0;

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && ix + stride < nx) {
            g_idata[idx] += g_idata[idx + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.y * gridDim.x + blockIdx.x] = g_idata[idx];
}
int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] %s starting transpose at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    /* host code */
    char in_filename[100];
    sprintf(in_filename, "%s.in", argv[1]);
    freopen(in_filename, "r", stdin);

    int n, k; cin >> n >> k;
    vector<vector<int>> A(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    int h_answer = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i][j] == k) h_answer++;
        }
    }
    printf("[host] host answer : %d\n", h_answer);

    /* host - device code */
    int nx = n, ny = n, nxy = n * n;
    int nbytes = nxy * sizeof(int);
    int* h_A = (int*)malloc(nbytes);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = A[i][j];
        }
    }

    /* device code */
    int* d_A, *d_odata;
    dim3 block(32, 1);
    dim3 grid((nx + block.x - 1) / block.x, ny);
    CHECK(cudaMalloc((void**)&d_A, nbytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * grid.y * sizeof(int)));
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    sumArrayElementK2D1D << <grid, block >> > (d_A, d_odata, nx, ny, k);
    int* h_odata = (int*)malloc(grid.x * grid.y * sizeof(int));
    cudaMemcpy(h_odata, d_odata, grid.x * grid.y * sizeof(int), cudaMemcpyDeviceToHost);
    int d_answer = 0;
    for (int i = 0; i < grid.x; i++) {
        for (int j = 0; j < grid.y; j++) {
            d_answer += h_odata[j * grid.x + i];
        }
    }
    printf("[host] device answer : %d\n", d_answer);
    checkResultInt(&h_answer, &d_answer, 1);

    // memory free
    free(h_A); free(h_odata);
    cudaFree(d_A); cudaFree(d_odata);
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==3036== NVPROF is profiling process 3036, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 3308
[host] datasize (40000), grid(4, 100), block(32, 1)
[host] device answer : 3308
[host] Arrays match.

==3036== Profiling application: ./Cuda.exe 1
==3036== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==3036== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.47%  15.456us         1  15.456us  15.456us  15.456us  sumArrayElementK2D1D(int*, int*, unsigned int, unsigned int, int)
                   25.62%  6.2400us         1  6.2400us  6.2400us  6.2400us  [CUDA memcpy HtoD]
                   10.91%  2.6560us         1  2.6560us  2.6560us  2.6560us  [CUDA memcpy DtoH]
      API calls:   68.31%  69.690ms         1  69.690ms  69.690ms  69.690ms  cudaSetDevice
                   28.96%  29.544ms         1  29.544ms  29.544ms  29.544ms  cuDevicePrimaryCtxRelease
                    1.35%  1.3768ms         1  1.3768ms  1.3768ms  1.3768ms  cudaLaunchKernel
                    0.71%  722.50us         2  361.25us  14.100us  708.40us  cudaFree
                    0.28%  288.90us         2  144.45us  4.8000us  284.10us  cudaMalloc
                    0.23%  238.80us         2  119.40us  66.700us  172.10us  cudaMemcpy
                    0.13%  133.80us         1  133.80us  133.80us  133.80us  cuLibraryUnload
                    0.02%  18.600us       114     163ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.01%  5.5000us         1  5.5000us  5.5000us  5.5000us  cudaGetDeviceProperties
                    0.00%  2.4000us         3     800ns     100ns  2.1000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==20380== NVPROF is profiling process 20380, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 3419
[host] datasize (40000), grid(4, 100), block(32, 1)
[host] device answer : 3419
[host] Arrays match.

==20380== Profiling application: ./Cuda.exe 2
==20380== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==20380== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.06%  15.808us         1  15.808us  15.808us  15.808us  sumArrayElementK2D1D(int*, int*, unsigned int, unsigned int, int)
                   29.27%  7.4560us         1  7.4560us  7.4560us  7.4560us  [CUDA memcpy HtoD]
                    8.67%  2.2080us         1  2.2080us  2.2080us  2.2080us  [CUDA memcpy DtoH]
      API calls:   73.92%  71.473ms         1  71.473ms  71.473ms  71.473ms  cudaSetDevice
                   23.57%  22.785ms         1  22.785ms  22.785ms  22.785ms  cuDevicePrimaryCtxRelease
                    1.22%  1.1776ms         1  1.1776ms  1.1776ms  1.1776ms  cudaLaunchKernel
                    0.65%  631.40us         2  315.70us  36.300us  595.10us  cudaFree
                    0.28%  270.60us         2  135.30us  26.500us  244.10us  cudaMalloc
                    0.22%  215.90us         2  107.95us  84.700us  131.20us  cudaMemcpy
                    0.08%  75.600us         1  75.600us  75.600us  75.600us  cuLibraryUnload
                    0.05%  43.600us       114     382ns       0ns  25.100us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuDeviceTotalMem
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuModuleGetLoadingMode
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.3000us         2     650ns     100ns  1.2000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==19832== NVPROF is profiling process 19832, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 3360
[host] datasize (40000), grid(4, 100), block(32, 1)
[host] device answer : 3360
[host] Arrays match.

==19832== Profiling application: ./Cuda.exe 3
==19832== Warning: 8 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==19832== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.46%  15.615us         1  15.615us  15.615us  15.615us  sumArrayElementK2D1D(int*, int*, unsigned int, unsigned int, int)
                   29.85%  7.5840us         1  7.5840us  7.5840us  7.5840us  [CUDA memcpy HtoD]
                    8.69%  2.2080us         1  2.2080us  2.2080us  2.2080us  [CUDA memcpy DtoH]
      API calls:   70.23%  79.358ms         1  79.358ms  79.358ms  79.358ms  cudaSetDevice
                   27.02%  30.537ms         1  30.537ms  30.537ms  30.537ms  cuDevicePrimaryCtxRelease
                    2.03%  2.2976ms         1  2.2976ms  2.2976ms  2.2976ms  cudaLaunchKernel
                    0.28%  316.40us         2  158.20us  32.600us  283.80us  cudaMalloc
                    0.20%  222.10us         2  111.05us  24.800us  197.30us  cudaFree
                    0.17%  192.60us         2  96.300us  93.300us  99.300us  cudaMemcpy
                    0.03%  33.500us         1  33.500us  33.500us  33.500us  cuLibraryUnload
                    0.02%  23.200us       114     203ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  5.4000us         3  1.8000us     200ns  2.6000us  cuDeviceGetCount
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.2000us         2     600ns     100ns  1.1000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
*/
