#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<bits/stdc++.h>
#include <time.h>
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
template<typename T>
void checkResult(T* h_data, T* d_data, const int n) {
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

__device__ __host__ int is_ok(int a) {
    int p = a % 10;
    a /= 10;
    if (p == 0) return 0;
    while (a) {
        int c = a % 10;
        a /= 10;
        if (c == 0 || abs(p - c) > 2) return 0;
        p = c;
    }
    return 1;
}

#define DIM 1024
__global__ void countConditialCase(int* h_A, int st, int ed) {
    int tid = threadIdx.x;
    int idx = st + blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int smem[DIM];

    // boundary check
    if (idx > ed) return;
    if (is_ok(idx)) smem[tid] = 1;
    else smem[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && idx+ stride <= ed) {
            smem[tid] += smem[tid + stride];
        }
    }
    if (tid == 0) h_A[blockIdx.x] = smem[0];
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

    int n; cin >> n;
    int *h_answer = (int *)malloc(sizeof(int));
    (*h_answer) = 0;

    clock_t start = clock();
    for (int i = (int)pow(10, n - 1); i < (int)pow(10, n); i++) {
        if (is_ok(i)) (*h_answer)++;
    }
    int times = 0;
    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] host time : %d ms\n", times);

    //printf("[host] host answer : ");
    //printf("%d", *h_answer);
   // printf("\n");

    /* host - device code */
    int st = (int)pow(10, n - 1), ed = (int)pow(10, n) - 1;
    int m = ed - st + 1;

    /* device code */
    start = clock();
    dim3 block(1024, 1);
    dim3 grid((m + block.x - 1) / block.x, 1);
    int* d_A;
    CHECK(cudaMalloc((void**)&d_A, grid.x * sizeof(int)));
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", m, grid.x, grid.y, block.x, block.y);
    countConditialCase << <grid, block >> > (d_A, st, ed);
    int* d_answer = (int*)malloc(sizeof(int));
    (*d_answer) = 0;
    int* tmp = (int*)malloc(grid.x * sizeof(int));
    CHECK(cudaMemcpy(tmp, d_A, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) (*d_answer) += tmp[i];

    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] device time : %d ms\n", times);
    
    //printf("[host] device answer : ");
   // printf("%d", *d_answer);
   // printf("\n");
    checkResult<int>(h_answer, d_answer, 1);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==20512== NVPROF is profiling process 20512, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (9), grid(1, 1), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==20512== Profiling application: ./Cuda.exe 1
==20512== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==20512== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   83.67%  11.808us         1  11.808us  11.808us  11.808us  countConditialCase(int*, int, int)
                   16.33%  2.3040us         1  2.3040us  2.3040us  2.3040us  [CUDA memcpy DtoH]
      API calls:   81.67%  116.53ms         1  116.53ms  116.53ms  116.53ms  cudaSetDevice
                   16.94%  24.169ms         1  24.169ms  24.169ms  24.169ms  cuDevicePrimaryCtxRelease
                    1.20%  1.7085ms         1  1.7085ms  1.7085ms  1.7085ms  cudaLaunchKernel
                    0.09%  130.00us         1  130.00us  130.00us  130.00us  cudaMalloc
                    0.06%  88.000us         1  88.000us  88.000us  88.000us  cudaMemcpy
                    0.02%  28.400us         1  28.400us  28.400us  28.400us  cuLibraryUnload
                    0.01%  18.700us       114     164ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  3.3000us         1  3.3000us  3.3000us  3.3000us  cudaGetDeviceProperties
                    0.00%  2.3000us         3     766ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==22900== NVPROF is profiling process 22900, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (90), grid(1, 1), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==22900== Profiling application: ./Cuda.exe 2
==22900== Warning: 22 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==22900== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.82%  15.392us         1  15.392us  15.392us  15.392us  countConditialCase(int*, int, int)
                   13.18%  2.3370us         1  2.3370us  2.3370us  2.3370us  [CUDA memcpy DtoH]
      API calls:   68.81%  64.697ms         1  64.697ms  64.697ms  64.697ms  cudaSetDevice
                   28.98%  27.252ms         1  27.252ms  27.252ms  27.252ms  cuDevicePrimaryCtxRelease
                    1.71%  1.6039ms         1  1.6039ms  1.6039ms  1.6039ms  cudaLaunchKernel
                    0.20%  188.60us         1  188.60us  188.60us  188.60us  cudaMemcpy
                    0.19%  182.10us         1  182.10us  182.10us  182.10us  cudaMalloc
                    0.07%  67.800us         1  67.800us  67.800us  67.800us  cuLibraryUnload
                    0.02%  23.200us       114     203ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.2000us         2     600ns     100ns  1.1000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==35580== NVPROF is profiling process 35580, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (900), grid(1, 1), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==35580== Profiling application: ./Cuda.exe 3
==35580== Warning: 35 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==35580== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   89.62%  20.448us         1  20.448us  20.448us  20.448us  countConditialCase(int*, int, int)
                   10.38%  2.3680us         1  2.3680us  2.3680us  2.3680us  [CUDA memcpy DtoH]
      API calls:   76.83%  82.362ms         1  82.362ms  82.362ms  82.362ms  cudaSetDevice
                   21.27%  22.799ms         1  22.799ms  22.799ms  22.799ms  cuDevicePrimaryCtxRelease
                    1.53%  1.6354ms         1  1.6354ms  1.6354ms  1.6354ms  cudaLaunchKernel
                    0.17%  184.70us         1  184.70us  184.70us  184.70us  cudaMalloc
                    0.15%  158.30us         1  158.30us  158.30us  158.30us  cudaMemcpy
                    0.03%  31.200us         1  31.200us  31.200us  31.200us  cuLibraryUnload
                    0.02%  18.100us       114     158ns       0ns  2.4000us  cuDeviceGetAttribute
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cudaGetDeviceProperties
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%     700ns         2     350ns     100ns     600ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 4
==23292== NVPROF is profiling process 23292, command: ./Cuda.exe 4
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (9000), grid(9, 1), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==23292== Profiling application: ./Cuda.exe 4
==23292== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==23292== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   90.35%  23.680us         1  23.680us  23.680us  23.680us  countConditialCase(int*, int, int)
                    9.65%  2.5280us         1  2.5280us  2.5280us  2.5280us  [CUDA memcpy DtoH]
      API calls:   69.20%  67.038ms         1  67.038ms  67.038ms  67.038ms  cudaSetDevice
                   28.98%  28.069ms         1  28.069ms  28.069ms  28.069ms  cuDevicePrimaryCtxRelease
                    1.25%  1.2082ms         1  1.2082ms  1.2082ms  1.2082ms  cudaLaunchKernel
                    0.24%  236.40us         1  236.40us  236.40us  236.40us  cudaMemcpy
                    0.16%  152.20us         1  152.20us  152.20us  152.20us  cudaMalloc
                    0.13%  124.20us         1  124.20us  124.20us  124.20us  cuLibraryUnload
                    0.03%  33.700us       114     295ns       0ns  15.100us  cuDeviceGetAttribute
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cudaGetDeviceProperties
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.8000us         3     600ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 5
==30876== NVPROF is profiling process 30876, command: ./Cuda.exe 5
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 5 ms
[host] datasize (90000), grid(88, 1), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==30876== Profiling application: ./Cuda.exe 5
==30876== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==30876== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.68%  162.59us         1  162.59us  162.59us  162.59us  countConditialCase(int*, int, int)
                    1.32%  2.1760us         1  2.1760us  2.1760us  2.1760us  [CUDA memcpy DtoH]
      API calls:   71.81%  65.717ms         1  65.717ms  65.717ms  65.717ms  cudaSetDevice
                   25.98%  23.777ms         1  23.777ms  23.777ms  23.777ms  cuDevicePrimaryCtxRelease
                    1.55%  1.4163ms         1  1.4163ms  1.4163ms  1.4163ms  cudaLaunchKernel
                    0.30%  270.70us         1  270.70us  270.70us  270.70us  cudaMemcpy
                    0.29%  269.20us         1  269.20us  269.20us  269.20us  cudaMalloc
                    0.04%  36.600us         1  36.600us  36.600us  36.600us  cuLibraryUnload
                    0.02%  20.400us       114     178ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.00%  3.2000us         1  3.2000us  3.2000us  3.2000us  cudaGetDeviceProperties
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.9000us         3     633ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 6
==22800== NVPROF is profiling process 22800, command: ./Cuda.exe 6
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 37 ms
[host] datasize (900000), grid(879, 1), block(1024, 1)
[host] device time : 3 ms
[host] Arrays match.

==22800== Profiling application: ./Cuda.exe 6
==22800== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==22800== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.85%  1.4650ms         1  1.4650ms  1.4650ms  1.4650ms  countConditialCase(int*, int, int)
                    0.15%  2.2400us         1  2.2400us  2.2400us  2.2400us  [CUDA memcpy DtoH]
      API calls:   70.25%  65.119ms         1  65.119ms  65.119ms  65.119ms  cudaSetDevice
                   26.28%  24.360ms         1  24.360ms  24.360ms  24.360ms  cuDevicePrimaryCtxRelease
                    1.73%  1.6068ms         1  1.6068ms  1.6068ms  1.6068ms  cudaMemcpy
                    1.30%  1.2024ms         1  1.2024ms  1.2024ms  1.2024ms  cudaLaunchKernel
                    0.35%  326.90us         1  326.90us  326.90us  326.90us  cudaMalloc
                    0.04%  34.800us       114     305ns       0ns  14.400us  cuDeviceGetAttribute
                    0.03%  31.600us         1  31.600us  31.600us  31.600us  cuLibraryUnload
                    0.00%  3.2000us         1  3.2000us  3.2000us  3.2000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 7
==5236== NVPROF is profiling process 5236, command: ./Cuda.exe 7
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 357 ms
[host] datasize (9000000), grid(8790, 1), block(1024, 1)
[host] device time : 16 ms
[host] Arrays match.

==5236== Profiling application: ./Cuda.exe 7
==5236== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==5236== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.93%  14.497ms         1  14.497ms  14.497ms  14.497ms  countConditialCase(int*, int, int)
                    0.07%  10.368us         1  10.368us  10.368us  10.368us  [CUDA memcpy DtoH]
      API calls:   62.76%  72.192ms         1  72.192ms  72.192ms  72.192ms  cudaSetDevice
                   23.13%  26.605ms         1  26.605ms  26.605ms  26.605ms  cuDevicePrimaryCtxRelease
                   12.75%  14.665ms         1  14.665ms  14.665ms  14.665ms  cudaMemcpy
                    1.09%  1.2517ms         1  1.2517ms  1.2517ms  1.2517ms  cudaLaunchKernel
                    0.21%  242.80us         1  242.80us  242.80us  242.80us  cudaMalloc
                    0.04%  42.100us         1  42.100us  42.100us  42.100us  cuLibraryUnload
                    0.02%  18.900us       114     165ns       0ns  2.4000us  cuDeviceGetAttribute
                    0.00%  3.4000us         1  3.4000us  3.4000us  3.4000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 8
==34184== NVPROF is profiling process 34184, command: ./Cuda.exe 8
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 3648 ms
[host] datasize (90000000), grid(87891, 1), block(1024, 1)
[host] device time : 148 ms
[host] Arrays match.

==34184== Profiling application: ./Cuda.exe 8
==34184== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==34184== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.93%  145.06ms         1  145.06ms  145.06ms  145.06ms  countConditialCase(int*, int, int)
                    0.07%  108.48us         1  108.48us  108.48us  108.48us  [CUDA memcpy DtoH]
      API calls:   62.23%  145.32ms         1  145.32ms  145.32ms  145.32ms  cudaMemcpy
                   25.36%  59.217ms         1  59.217ms  59.217ms  59.217ms  cudaSetDevice
                   11.43%  26.689ms         1  26.689ms  26.689ms  26.689ms  cuDevicePrimaryCtxRelease
                    0.79%  1.8347ms         1  1.8347ms  1.8347ms  1.8347ms  cudaLaunchKernel
                    0.16%  368.00us         1  368.00us  368.00us  368.00us  cudaMalloc
                    0.02%  51.600us         1  51.600us  51.600us  51.600us  cuLibraryUnload
                    0.01%  18.100us       114     158ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         3     666ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 9
==24724== NVPROF is profiling process 24724, command: ./Cuda.exe 9
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 36954 ms
[host] datasize (900000000), grid(878907, 1), block(1024, 1)
[host] device time : 1413 ms
[host] Arrays match.

==24724== Profiling application: ./Cuda.exe 9
==24724== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24724== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.92%  1.40670s         1  1.40670s  1.40670s  1.40670s  countConditialCase(int*, int, int)
                    0.08%  1.0935ms         1  1.0935ms  1.0935ms  1.0935ms  [CUDA memcpy DtoH]
      API calls:   94.09%  1.40841s         1  1.40841s  1.40841s  1.40841s  cudaMemcpy
                    3.94%  59.029ms         1  59.029ms  59.029ms  59.029ms  cudaSetDevice
                    1.80%  26.985ms         1  26.985ms  26.985ms  26.985ms  cuDevicePrimaryCtxRelease
                    0.14%  2.0789ms         1  2.0789ms  2.0789ms  2.0789ms  cudaLaunchKernel
                    0.02%  232.60us         1  232.60us  232.60us  232.60us  cudaMalloc
                    0.00%  51.800us         1  51.800us  51.800us  51.800us  cuLibraryUnload
                    0.00%  38.400us       114     336ns       0ns  16.600us  cuDeviceGetAttribute
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
