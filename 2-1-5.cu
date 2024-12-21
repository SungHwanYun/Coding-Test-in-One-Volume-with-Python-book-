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

__global__ void sumArrayElementK2D1D(int* g_idata, int* g_odata, unsigned int nx, unsigned int ny, 
    int x1, int y1, int x2, int y2, int k) {
    unsigned int tid = threadIdx.x;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // blockDim.y(=1), threadIdx.y(=0)
    unsigned int idx = iy * nx + ix;

    // boundary check
    if (ix >= nx || iy >= ny) return;

    // branch divergence!!!
    if (x1 <= ix && ix <= x2 && y1 <= iy && iy <= y2) g_idata[idx] *= k;

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

    int n; cin >> n;
    vector<vector<int>> A(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    int r1, c1, r2, c2, k; cin >> r1 >> c1 >> r2 >> c2 >> k;
    int h_answer = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (r1 <= i && i <= r2 && c1 <= j && j <= c2) h_answer += A[i][j] * k;
            else h_answer += A[i][j];
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
    sumArrayElementK2D1D << <grid, block >> > (d_A, d_odata, nx, ny, c1, r1, c2, r2, k);
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
==30300== NVPROF is profiling process 30300, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 51680667
[host] datasize (40000), grid(4, 100), block(32, 1)
[host] device answer : 51680667
[host] Arrays match.

==30300== Profiling application: ./Cuda.exe 1
==30300== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==30300== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.71%  16.192us         1  16.192us  16.192us  16.192us  sumArrayElementK2D1D(int*, int*, unsigned int, unsigned int, int, int, int, int, int)
                   25.20%  6.2090us         1  6.2090us  6.2090us  6.2090us  [CUDA memcpy HtoD]
                    9.09%  2.2400us         1  2.2400us  2.2400us  2.2400us  [CUDA memcpy DtoH]
      API calls:   77.66%  111.58ms         1  111.58ms  111.58ms  111.58ms  cudaSetDevice
                   20.29%  29.152ms         1  29.152ms  29.152ms  29.152ms  cuDevicePrimaryCtxRelease
                    1.32%  1.9012ms         1  1.9012ms  1.9012ms  1.9012ms  cudaLaunchKernel
                    0.25%  353.80us         2  176.90us  116.00us  237.80us  cudaMemcpy
                    0.25%  353.20us         2  176.60us  15.500us  337.70us  cudaFree
                    0.20%  289.60us         2  144.80us  6.4000us  283.20us  cudaMalloc
                    0.02%  24.900us         1  24.900us  24.900us  24.900us  cuLibraryUnload
                    0.01%  19.600us       114     171ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  4.3000us         1  4.3000us  4.3000us  4.3000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  1.9000us         3     633ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==12476== NVPROF is profiling process 12476, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 198317317
[host] datasize (40000), grid(4, 100), block(32, 1)
[host] device answer : 198317317
[host] Arrays match.

==12476== Profiling application: ./Cuda.exe 2
==12476== Warning: 39 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==12476== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.33%  16.256us         1  16.256us  16.256us  16.256us  sumArrayElementK2D1D(int*, int*, unsigned int, unsigned int, int, int, int, int, int)
                   28.34%  7.3920us         1  7.3920us  7.3920us  7.3920us  [CUDA memcpy HtoD]
                    9.33%  2.4320us         1  2.4320us  2.4320us  2.4320us  [CUDA memcpy DtoH]
      API calls:   73.32%  75.654ms         1  75.654ms  75.654ms  75.654ms  cudaSetDevice
                   23.84%  24.602ms         1  24.602ms  24.602ms  24.602ms  cuDevicePrimaryCtxRelease
                    2.09%  2.1571ms         1  2.1571ms  2.1571ms  2.1571ms  cudaLaunchKernel
                    0.28%  287.20us         2  143.60us  5.8000us  281.40us  cudaMalloc
                    0.22%  225.70us         2  112.85us  96.400us  129.30us  cudaMemcpy
                    0.18%  187.40us         2  93.700us  20.500us  166.90us  cudaFree
                    0.03%  32.100us         1  32.100us  32.100us  32.100us  cuLibraryUnload
                    0.02%  19.100us       114     167ns       0ns  3.4000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetLuid
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==26852== NVPROF is profiling process 26852, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 60705528
[host] datasize (40000), grid(4, 100), block(32, 1)
[host] device answer : 60705528
[host] Arrays match.

==26852== Profiling application: ./Cuda.exe 3
==26852== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==26852== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.53%  16.000us         1  16.000us  16.000us  16.000us  sumArrayElementK2D1D(int*, int*, unsigned int, unsigned int, int, int, int, int, int)
                   26.30%  6.6240us         1  6.6240us  6.6240us  6.6240us  [CUDA memcpy HtoD]
                   10.17%  2.5600us         1  2.5600us  2.5600us  2.5600us  [CUDA memcpy DtoH]
      API calls:   71.57%  70.582ms         1  70.582ms  70.582ms  70.582ms  cudaSetDevice
                   25.41%  25.058ms         1  25.058ms  25.058ms  25.058ms  cuDevicePrimaryCtxRelease
                    2.12%  2.0945ms         1  2.0945ms  2.0945ms  2.0945ms  cudaLaunchKernel
                    0.28%  279.60us         2  139.80us  6.8000us  272.80us  cudaMalloc
                    0.27%  264.80us         2  132.40us  27.000us  237.80us  cudaFree
                    0.26%  259.20us         2  129.60us  118.40us  140.80us  cudaMemcpy
                    0.05%  44.700us         1  44.700us  44.700us  44.700us  cuLibraryUnload
                    0.02%  20.500us       114     179ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  3.0000us         1  3.0000us  3.0000us  3.0000us  cudaGetDeviceProperties
                    0.00%  2.0000us         3     666ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
