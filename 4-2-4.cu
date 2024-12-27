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
            printf("[host] host %5lld gpu %5lld at current %d\n", h_data[i], d_data[i], i);
            break;
        }
    }
    if (match) printf("[host] Arrays match.\n\n");
}

__device__ __host__ int is_ok(long long a) {
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
__global__ void countConditialCase(int* h_A, int nx, int ny, long long st, long long ed) {
    int tid = threadIdx.x;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    long long idx = st + (long long)iy * nx + ix;
    __shared__ int smem[DIM];

    // boundary check
    if (ix >=nx || iy >=ny || idx > ed ) return;
    if (is_ok(idx)) smem[tid] = 1;
    else smem[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && idx+ stride <= ed) {
            smem[tid] += smem[tid + stride];
        }
    }
    if (tid == 0) h_A[blockIdx.y * gridDim.x + blockIdx.x] = smem[0];
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
    long long*h_answer = (long long*)malloc(sizeof(long long));
    (*h_answer) = 0;

    clock_t start = clock();
    for (long long i = (long long)pow(10, n - 1); i < (long long)pow(10, n); i++) {
        if (is_ok(i)) (*h_answer)++;
    }
    int times = 0;
    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] host time : %d ms\n", times);

    //printf("[host] host answer : ");
    //printf("%d", *h_answer);
   // printf("\n");

    /* host - device code */
    long long st = (long long)pow(10, n - 1), ed = (long long)pow(10, n) - 1;
    long long m = ed - st + 1;

    /* device code */
    start = clock();
    dim3 block(1024, 1);
    int k = (m + block.x - 1) / block.x;
    int nx = (int)sqrt(k), ny = (k + nx - 1) / nx;
    dim3 grid(nx, ny);
    int* d_A;
    CHECK(cudaMalloc((void**)&d_A, grid.x * grid.y * sizeof(int)));
    printf("[host] datasize (%lld), grid(%d, %d), block(%d, %d)\n", m, grid.x, grid.y, block.x, block.y);
    countConditialCase << <grid, block >> > (d_A, nx * block.x, ny * block.y, st, ed);
    long long* d_answer = (long long*)malloc(sizeof(long long));
    (*d_answer) = 0;
    int* tmp = (int*)malloc(grid.x * grid.y * sizeof(int));
    CHECK(cudaMemcpy(tmp, d_A, grid.x * grid.y * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid.x; i++) for (int j = 0; j < grid.y; j++) {
        (*d_answer) += tmp[j * grid.x + i];
    }

    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] device time : %d ms\n", times);
    
    //printf("[host] device answer : ");
   // printf("%d", *d_answer);
   // printf("\n");
    checkResult<long long>(h_answer, d_answer, 1);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==36004== NVPROF is profiling process 36004, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (9), grid(1, 1), block(1024, 1)
[host] device time : 1 ms
[host] Arrays match.

==36004== Profiling application: ./Cuda.exe 1
==36004== Warning: 25 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==36004== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   88.63%  18.208us         1  18.208us  18.208us  18.208us  countConditialCase(int*, int, int, __int64, __int64)
                   11.37%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
      API calls:   78.36%  100.19ms         1  100.19ms  100.19ms  100.19ms  cudaSetDevice
                   20.66%  26.413ms         1  26.413ms  26.413ms  26.413ms  cuDevicePrimaryCtxRelease
                    0.72%  921.70us         1  921.70us  921.70us  921.70us  cudaLaunchKernel
                    0.13%  165.00us         1  165.00us  165.00us  165.00us  cudaMalloc
                    0.07%  84.600us         1  84.600us  84.600us  84.600us  cudaMemcpy
                    0.04%  45.100us         1  45.100us  45.100us  45.100us  cuLibraryUnload
                    0.01%  18.500us       114     162ns       0ns  2.2000us  cuDeviceGetAttribute
                    0.01%  6.6000us         1  6.6000us  6.6000us  6.6000us  cudaGetDeviceProperties
                    0.00%  2.7000us         3     900ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==35164== NVPROF is profiling process 35164, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (90), grid(1, 1), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==35164== Profiling application: ./Cuda.exe 2
==35164== Warning: 17 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==35164== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   89.20%  25.632us         1  25.632us  25.632us  25.632us  countConditialCase(int*, int, int, __int64, __int64)
                   10.80%  3.1040us         1  3.1040us  3.1040us  3.1040us  [CUDA memcpy DtoH]
      API calls:   75.46%  81.678ms         1  81.678ms  81.678ms  81.678ms  cudaSetDevice
                   23.13%  25.034ms         1  25.034ms  25.034ms  25.034ms  cuDevicePrimaryCtxRelease
                    1.04%  1.1276ms         1  1.1276ms  1.1276ms  1.1276ms  cudaLaunchKernel
                    0.14%  149.00us         1  149.00us  149.00us  149.00us  cudaMalloc
                    0.14%  147.30us         1  147.30us  147.30us  147.30us  cudaMemcpy
                    0.07%  73.500us         1  73.500us  73.500us  73.500us  cuLibraryUnload
                    0.02%  23.300us       114     204ns       0ns  3.5000us  cuDeviceGetAttribute
                    0.00%  4.4000us         1  4.4000us  4.4000us  4.4000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.3000us         3     766ns       0ns  2.0000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==13320== NVPROF is profiling process 13320, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (900), grid(1, 1), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==13320== Profiling application: ./Cuda.exe 3
==13320== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==13320== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.98%  37.440us         1  37.440us  37.440us  37.440us  countConditialCase(int*, int, int, __int64, __int64)
                    6.02%  2.4000us         1  2.4000us  2.4000us  2.4000us  [CUDA memcpy DtoH]
      API calls:   69.36%  68.544ms         1  68.544ms  68.544ms  68.544ms  cudaSetDevice
                   28.90%  28.556ms         1  28.556ms  28.556ms  28.556ms  cuDevicePrimaryCtxRelease
                    1.15%  1.1383ms         1  1.1383ms  1.1383ms  1.1383ms  cudaLaunchKernel
                    0.23%  224.10us         1  224.10us  224.10us  224.10us  cuLibraryUnload
                    0.18%  176.10us         1  176.10us  176.10us  176.10us  cudaMemcpy
                    0.15%  147.00us         1  147.00us  147.00us  147.00us  cudaMalloc
                    0.02%  19.600us       114     171ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  4.4000us         1  4.4000us  4.4000us  4.4000us  cudaGetDeviceProperties
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         3     633ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 4
==27816== NVPROF is profiling process 27816, command: ./Cuda.exe 4
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (9000), grid(3, 3), block(1024, 1)
[host] device time : 2 ms
[host] Arrays match.

==27816== Profiling application: ./Cuda.exe 4
==27816== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==27816== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   95.10%  47.776us         1  47.776us  47.776us  47.776us  countConditialCase(int*, int, int, __int64, __int64)
                    4.90%  2.4640us         1  2.4640us  2.4640us  2.4640us  [CUDA memcpy DtoH]
      API calls:   71.65%  67.652ms         1  67.652ms  67.652ms  67.652ms  cudaSetDevice
                   26.20%  24.740ms         1  24.740ms  24.740ms  24.740ms  cuDevicePrimaryCtxRelease
                    1.47%  1.3905ms         1  1.3905ms  1.3905ms  1.3905ms  cudaLaunchKernel
                    0.28%  261.30us         1  261.30us  261.30us  261.30us  cudaMemcpy
                    0.22%  204.70us         1  204.70us  204.70us  204.70us  cudaMalloc
                    0.15%  140.80us         1  140.80us  140.80us  140.80us  cuLibraryUnload
                    0.02%  18.200us       114     159ns       0ns  2.3000us  cuDeviceGetAttribute
                    0.00%  4.4000us         1  4.4000us  4.4000us  4.4000us  cudaGetDeviceProperties
                    0.00%  2.5000us         3     833ns     100ns  2.0000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 5
==35856== NVPROF is profiling process 35856, command: ./Cuda.exe 5
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 4 ms
[host] datasize (90000), grid(9, 10), block(1024, 1)
[host] device time : 1 ms
[host] Arrays match.

==35856== Profiling application: ./Cuda.exe 5
==35856== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==35856== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.33%  351.17us         1  351.17us  351.17us  351.17us  countConditialCase(int*, int, int, __int64, __int64)
                    0.67%  2.3680us         1  2.3680us  2.3680us  2.3680us  [CUDA memcpy DtoH]
      API calls:   70.17%  61.536ms         1  61.536ms  61.536ms  61.536ms  cudaSetDevice
                   27.68%  24.272ms         1  24.272ms  24.272ms  24.272ms  cuDevicePrimaryCtxRelease
                    1.18%  1.0368ms         1  1.0368ms  1.0368ms  1.0368ms  cudaLaunchKernel
                    0.51%  443.80us         1  443.80us  443.80us  443.80us  cudaMemcpy
                    0.22%  197.00us         1  197.00us  197.00us  197.00us  cudaMalloc
                    0.20%  173.70us         1  173.70us  173.70us  173.70us  cuLibraryUnload
                    0.02%  19.100us       114     167ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.01%  4.5000us         1  4.5000us  4.5000us  4.5000us  cudaGetDeviceProperties
                    0.00%  2.9000us         3     966ns       0ns  2.6000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 6
==23152== NVPROF is profiling process 23152, command: ./Cuda.exe 6
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 44 ms
[host] datasize (900000), grid(29, 31), block(1024, 1)
[host] device time : 5 ms
[host] Arrays match.

==23152== Profiling application: ./Cuda.exe 6
==23152== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==23152== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.93%  3.2782ms         1  3.2782ms  3.2782ms  3.2782ms  countConditialCase(int*, int, int, __int64, __int64)
                    0.07%  2.2400us         1  2.2400us  2.2400us  2.2400us  [CUDA memcpy DtoH]
      API calls:   67.96%  71.604ms         1  71.604ms  71.604ms  71.604ms  cudaSetDevice
                   27.38%  28.847ms         1  28.847ms  28.847ms  28.847ms  cuDevicePrimaryCtxRelease
                    3.21%  3.3836ms         1  3.3836ms  3.3836ms  3.3836ms  cudaMemcpy
                    1.06%  1.1158ms         1  1.1158ms  1.1158ms  1.1158ms  cudaLaunchKernel
                    0.30%  318.90us         1  318.90us  318.90us  318.90us  cudaMalloc
                    0.05%  56.900us         1  56.900us  56.900us  56.900us  cuLibraryUnload
                    0.02%  18.200us       114     159ns       0ns  2.2000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cuModuleGetLoadingMode
                    0.00%  2.8000us         3     933ns     100ns  2.0000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.2000us         2     600ns     200ns  1.0000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 7
==34760== NVPROF is profiling process 34760, command: ./Cuda.exe 7
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 404 ms
[host] datasize (9000000), grid(93, 95), block(1024, 1)
[host] device time : 35 ms
[host] Arrays match.

==34760== Profiling application: ./Cuda.exe 7
==34760== Warning: 36 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==34760== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.97%  32.800ms         1  32.800ms  32.800ms  32.800ms  countConditialCase(int*, int, int, __int64, __int64)
                    0.03%  10.400us         1  10.400us  10.400us  10.400us  [CUDA memcpy DtoH]
      API calls:   51.61%  69.170ms         1  69.170ms  69.170ms  69.170ms  cudaSetDevice
                   24.68%  33.081ms         1  33.081ms  33.081ms  33.081ms  cudaMemcpy
                   22.53%  30.198ms         1  30.198ms  30.198ms  30.198ms  cuDevicePrimaryCtxRelease
                    0.91%  1.2212ms         1  1.2212ms  1.2212ms  1.2212ms  cudaLaunchKernel
                    0.19%  255.00us         1  255.00us  255.00us  255.00us  cudaMalloc
                    0.05%  60.700us         1  60.700us  60.700us  60.700us  cuLibraryUnload
                    0.01%  19.300us       114     169ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.8000us         3     600ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.4000us         2     700ns     100ns  1.3000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 8
==19772== NVPROF is profiling process 19772, command: ./Cuda.exe 8
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 3786 ms
[host] datasize (90000000), grid(296, 297), block(1024, 1)
[host] device time : 283 ms
[host] Arrays match.

==19772== Profiling application: ./Cuda.exe 8
==19772== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==19772== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.96%  280.85ms         1  280.85ms  280.85ms  280.85ms  countConditialCase(int*, int, int, __int64, __int64)
                    0.04%  108.48us         1  108.48us  108.48us  108.48us  [CUDA memcpy DtoH]
      API calls:   74.19%  281.15ms         1  281.15ms  281.15ms  281.15ms  cudaMemcpy
                   17.54%  66.473ms         1  66.473ms  66.473ms  66.473ms  cudaSetDevice
                    7.83%  29.688ms         1  29.688ms  29.688ms  29.688ms  cuDevicePrimaryCtxRelease
                    0.33%  1.2648ms         1  1.2648ms  1.2648ms  1.2648ms  cudaLaunchKernel
                    0.07%  255.50us         1  255.50us  255.50us  255.50us  cudaMalloc
                    0.02%  72.700us         1  72.700us  72.700us  72.700us  cuLibraryUnload
                    0.00%  18.700us       114     164ns       0ns  2.3000us  cuDeviceGetAttribute
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuModuleGetLoadingMode
                    0.00%  2.3000us         3     766ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 9
==37284== NVPROF is profiling process 37284, command: ./Cuda.exe 9
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 37731 ms
[host] datasize (900000000), grid(937, 939), block(1024, 1)
[host] device time : 2803 ms
[host] Arrays match.

==37284== Profiling application: ./Cuda.exe 9
==37284== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==37284== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.96%  2.79487s         1  2.79487s  2.79487s  2.79487s  countConditialCase(int*, int, int, __int64, __int64)
                    0.04%  1.0834ms         1  1.0834ms  1.0834ms  1.0834ms  [CUDA memcpy DtoH]
      API calls:   96.51%  2.79634s         1  2.79634s  2.79634s  2.79634s  cudaMemcpy
                    2.40%  69.628ms         1  69.628ms  69.628ms  69.628ms  cudaSetDevice
                    1.00%  29.044ms         1  29.044ms  29.044ms  29.044ms  cuDevicePrimaryCtxRelease
                    0.07%  2.1203ms         1  2.1203ms  2.1203ms  2.1203ms  cudaLaunchKernel
                    0.01%  228.60us         1  228.60us  228.60us  228.60us  cudaMalloc
                    0.00%  52.700us         1  52.700us  52.700us  52.700us  cuLibraryUnload
                    0.00%  18.300us       114     160ns       0ns  2.2000us  cuDeviceGetAttribute
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         3     633ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.3000us         2     650ns     200ns  1.1000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 10
==37652== NVPROF is profiling process 37652, command: ./Cuda.exe 10
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 380684 ms
[host] datasize (9000000000), grid(2964, 2966), block(1024, 1)
[host] device time : 60138 ms
[host] Arrays match.

==37652== Profiling application: ./Cuda.exe 10
==37652== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==37652== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.98%  60.0519s         1  60.0519s  60.0519s  60.0519s  countConditialCase(int*, int, int, __int64, __int64)
                    0.02%  11.001ms         1  11.001ms  11.001ms  11.001ms  [CUDA memcpy DtoH]
      API calls:   99.77%  60.0588s         1  60.0588s  60.0588s  60.0588s  cudaMemcpy
                    0.15%  88.512ms         1  88.512ms  88.512ms  88.512ms  cudaSetDevice
                    0.06%  33.468ms         1  33.468ms  33.468ms  33.468ms  cuDevicePrimaryCtxRelease
                    0.03%  16.495ms         1  16.495ms  16.495ms  16.495ms  cudaLaunchKernel
                    0.00%  309.00us         1  309.00us  309.00us  309.00us  cudaMalloc
                    0.00%  89.400us         1  89.400us  89.400us  89.400us  cuLibraryUnload
                    0.00%  19.200us       114     168ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.9000us         3     633ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
