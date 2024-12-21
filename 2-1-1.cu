#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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

__global__ void sumArrayElementK(int* g_idata, int* g_odata, unsigned int n, int k) {
    unsigned int tid = threadIdx.x;
    int* idata = g_idata + blockIdx.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    // branch divergence!!!
    if (g_idata[idx] == k) g_idata[idx] = 1;
    else g_idata[idx] = 0;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] %s starting transpose at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    char in_filename[100];
    sprintf(in_filename, "%s.in", argv[1]);
    freopen(in_filename, "r", stdin);

    int n, k;
    scanf("%d %d", &n, &k);
    int nbytes = n * sizeof(int);
    int* h_A = (int*)malloc(nbytes);
    for (int i = 0; i < n; i++) {
        scanf("%d", h_A + i);
    }

    int h_answer = 0;
    for (int i = 0; i < n; i++) {
        if (h_A[i] == k) h_answer++;
    }
    printf("[host] host answer : %d\n", h_answer);

    int* d_idata, * d_odata;
    int blocksize = 512;
    int size = n;
    dim3 block(blocksize, 1);
    dim3 grid((nbytes + blocksize - 1) / blocksize, 1);
    CHECK(cudaMalloc((void**)&d_idata, nbytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));
    cudaMemcpy(d_idata, h_A, nbytes, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), gird(%d), block(%d)\n", size, grid.x, block.x);
    sumArrayElementK << <grid, block >> > (d_idata, d_odata, n, k);
    int* h_odata = (int*)malloc(grid.x * sizeof(int));
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    int d_answer = 0;
    for (int i = 0; i < grid.x; i++) d_answer += h_odata[i];
    printf("[host] device answer : %d\n", d_answer);
    checkResultInt(&h_answer, &d_answer, 1);

    // memory free
    free(h_A);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==27652== NVPROF is profiling process 27652, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 368
[host] datasize (1000), gird(8), block(512)
[host] device answer : 368
[host] Arrays match.

==27652== Profiling application: ./Cuda.exe 1
==27652== Warning: 28 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==27652== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.96%  9.5040us         1  9.5040us  9.5040us  9.5040us  sumArrayElementK(int*, int*, unsigned int, int)
                   16.48%  2.3040us         1  2.3040us  2.3040us  2.3040us  [CUDA memcpy DtoH]
                   15.56%  2.1760us         1  2.1760us  2.1760us  2.1760us  [CUDA memcpy HtoD]
      API calls:   73.07%  70.262ms         1  70.262ms  70.262ms  70.262ms  cudaSetDevice
                   24.87%  23.915ms         1  23.915ms  23.915ms  23.915ms  cuDevicePrimaryCtxRelease
                    1.06%  1.0187ms         1  1.0187ms  1.0187ms  1.0187ms  cudaLaunchKernel
                    0.36%  344.10us         2  172.05us  9.3000us  334.80us  cudaMalloc
                    0.31%  295.90us         2  147.95us  29.300us  266.60us  cudaFree
                    0.18%  168.90us         2  84.450us  66.300us  102.60us  cudaMemcpy
                    0.12%  116.00us         1  116.00us  116.00us  116.00us  cuLibraryUnload
                    0.02%  22.000us       114     192ns       0ns  4.6000us  cuDeviceGetAttribute
                    0.01%  6.9000us         1  6.9000us  6.9000us  6.9000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.2000us         2     600ns     100ns  1.1000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==28504== NVPROF is profiling process 28504, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 371
[host] datasize (1000), gird(8), block(512)
[host] device answer : 371
[host] Arrays match.

==28504== Profiling application: ./Cuda.exe 2
==28504== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28504== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.95%  9.5680us         1  9.5680us  9.5680us  9.5680us  sumArrayElementK(int*, int*, unsigned int, int)
                   16.59%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
                   15.45%  2.1760us         1  2.1760us  2.1760us  2.1760us  [CUDA memcpy HtoD]
      API calls:   69.03%  66.941ms         1  66.941ms  66.941ms  66.941ms  cudaSetDevice
                   28.99%  28.114ms         1  28.114ms  28.114ms  28.114ms  cuDevicePrimaryCtxRelease
                    1.03%  1.0035ms         1  1.0035ms  1.0035ms  1.0035ms  cudaLaunchKernel
                    0.41%  397.60us         2  198.80us  9.3000us  388.30us  cudaFree
                    0.24%  231.20us         2  115.60us  5.2000us  226.00us  cudaMalloc
                    0.17%  167.40us         2  83.700us  58.100us  109.30us  cudaMemcpy
                    0.04%  42.000us         1  42.000us  42.000us  42.000us  cuLibraryUnload
                    0.03%  33.600us       114     294ns       0ns  7.0000us  cuDeviceGetAttribute
                    0.03%  25.800us         1  25.800us  25.800us  25.800us  cudaGetDeviceProperties
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cuDeviceGetUuid
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  1.9000us         3     633ns       0ns  1.6000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==9228== NVPROF is profiling process 9228, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 349
[host] datasize (1000), gird(8), block(512)
[host] device answer : 349
[host] Arrays match.

==9228== Profiling application: ./Cuda.exe 3
==9228== Warning: 1 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==9228== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.28%  9.8880us         1  9.8880us  9.8880us  9.8880us  sumArrayElementK(int*, int*, unsigned int, int)
                   15.70%  2.2400us         1  2.2400us  2.2400us  2.2400us  [CUDA memcpy DtoH]
                   15.02%  2.1440us         1  2.1440us  2.1440us  2.1440us  [CUDA memcpy HtoD]
      API calls:   75.98%  81.343ms         1  81.343ms  81.343ms  81.343ms  cudaSetDevice
                   21.76%  23.295ms         1  23.295ms  23.295ms  23.295ms  cuDevicePrimaryCtxRelease
                    0.98%  1.0530ms         1  1.0530ms  1.0530ms  1.0530ms  cudaLaunchKernel
                    0.58%  623.40us         2  311.70us  15.100us  608.30us  cudaFree
                    0.31%  328.00us         2  164.00us  23.200us  304.80us  cudaMalloc
                    0.26%  276.40us         2  138.20us  65.300us  211.10us  cudaMemcpy
                    0.09%  100.70us         1  100.70us  100.70us  100.70us  cuLibraryUnload
                    0.03%  27.600us       114     242ns       0ns  3.9000us  cuDeviceGetAttribute
                    0.00%  4.5000us         1  4.5000us  4.5000us  4.5000us  cudaGetDeviceProperties
                    0.00%  3.7000us         3  1.2330us     100ns  3.3000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
