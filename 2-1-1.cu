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
    printf("[host] answer : %d\n", h_answer);

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
    printf("[host] answer : %d\n", d_answer);
    checkResultInt(&h_answer, &d_answer, 1);
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof Cuda.exe 1
==27936== NVPROF is profiling process 27936, command: Cuda.exe 1
[host] Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] answer : 368
[host] datasize (1000), gird(8), block(512)
[host] answer : 368
[host] Arrays match.

==27936== Profiling application: Cuda.exe 1
==27936== Warning: 12 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==27936== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   66.45%  9.5050us         1  9.5050us  9.5050us  9.5050us  sumArrayElementK(int*, int*, unsigned int, int)
                   17.89%  2.5590us         1  2.5590us  2.5590us  2.5590us  [CUDA memcpy DtoH]
                   15.66%  2.2400us         1  2.2400us  2.2400us  2.2400us  [CUDA memcpy HtoD]
      API calls:   72.26%  65.873ms         1  65.873ms  65.873ms  65.873ms  cudaSetDevice
                   25.90%  23.613ms         1  23.613ms  23.613ms  23.613ms  cuDevicePrimaryCtxRelease
                    1.18%  1.0730ms         1  1.0730ms  1.0730ms  1.0730ms  cudaLaunchKernel
                    0.25%  230.60us         2  115.30us  49.100us  181.50us  cudaMemcpy
                    0.22%  204.80us         2  102.40us  4.9000us  199.90us  cudaMalloc
                    0.13%  119.40us         1  119.40us  119.40us  119.40us  cuLibraryUnload
                    0.03%  28.700us       114     251ns       0ns  4.8000us  cuDeviceGetAttribute
                    0.00%  4.5000us         1  4.5000us  4.5000us  4.5000us  cudaGetDeviceProperties
                    0.00%  3.2000us         1  3.2000us  3.2000us  3.2000us  cuDeviceTotalMem
                    0.00%  2.2000us         3     733ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof Cuda.exe 2
==6444== NVPROF is profiling process 6444, command: Cuda.exe 2
[host] Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] answer : 371
[host] datasize (1000), gird(8), block(512)
[host] answer : 371
[host] Arrays match.

==6444== Profiling application: Cuda.exe 2
==6444== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==6444== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.78%  9.8240us         1  9.8240us  9.8240us  9.8240us  sumArrayElementK(int*, int*, unsigned int, int)
                   16.34%  2.3680us         1  2.3680us  2.3680us  2.3680us  [CUDA memcpy DtoH]
                   15.89%  2.3030us         1  2.3030us  2.3030us  2.3030us  [CUDA memcpy HtoD]
      API calls:   66.71%  67.935ms         1  67.935ms  67.935ms  67.935ms  cudaSetDevice
                   31.79%  32.376ms         1  32.376ms  32.376ms  32.376ms  cuDevicePrimaryCtxRelease
                    1.00%  1.0181ms         1  1.0181ms  1.0181ms  1.0181ms  cudaLaunchKernel
                    0.24%  241.70us         2  120.85us  5.0000us  236.70us  cudaMalloc
                    0.16%  160.60us         2  80.300us  58.300us  102.30us  cudaMemcpy
                    0.07%  73.600us         1  73.600us  73.600us  73.600us  cuLibraryUnload
                    0.02%  18.300us       114     160ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaGetDeviceProperties
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof Cuda.exe 3
==28844== NVPROF is profiling process 28844, command: Cuda.exe 3
[host] Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] answer : 349
[host] datasize (1000), gird(8), block(512)
[host] answer : 349
[host] Arrays match.

==28844== Profiling application: Cuda.exe 3
==28844== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28844== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.27%  9.5360us         1  9.5360us  9.5360us  9.5360us  sumArrayElementK(int*, int*, unsigned int, int)
                   16.48%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
                   16.25%  2.3040us         1  2.3040us  2.3040us  2.3040us  [CUDA memcpy HtoD]
      API calls:   72.64%  66.181ms         1  66.181ms  66.181ms  66.181ms  cudaSetDevice
                   26.03%  23.715ms         1  23.715ms  23.715ms  23.715ms  cuDevicePrimaryCtxRelease
                    0.87%  796.60us         1  796.60us  796.60us  796.60us  cudaLaunchKernel
                    0.22%  196.40us         2  98.200us  6.9000us  189.50us  cudaMalloc
                    0.14%  131.90us         2  65.950us  58.500us  73.400us  cudaMemcpy
                    0.06%  52.300us         1  52.300us  52.300us  52.300us  cuLibraryUnload
                    0.02%  17.700us       114     155ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  1.9000us         3     633ns       0ns  1.6000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
