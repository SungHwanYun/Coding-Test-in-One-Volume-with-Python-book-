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

__global__ void compareArrayElement(int* g_A, int* g_B, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    int* iA = g_A + blockIdx.x * blockDim.x;
    int* iB = g_B + blockIdx.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    // branch divergence!!!
    if (iA[tid] > iB[tid]) iA[tid] = 1;
    else if (iA[tid] < iB[tid]) iA[tid] = -1;
    else iA[tid] = 0;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && idx + stride < n) {
            iA[tid] += iA[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = iA[0];
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

    vector<int> A;
    do {
        int a; cin >> a;
        A.push_back(a);
    } while (getc(stdin) == ' ');

    vector<int> B;
    do {
        int b; cin >> b;
        B.push_back(b);
    } while (getc(stdin) == ' ');

    int n = (int)A.size();
    int nbytes = n * sizeof(int);
    int a = 0, b = 0;
    for (int i = 0; i < n; i++) {
        if (A[i] > B[i]) a++;
        else if (A[i] < B[i]) b++;
    }
    int h_answer = 0;
    if (a > b) h_answer = 1;
    printf("[host] host answer : %d\n", h_answer);
    int* h_A, * h_B;
    h_A = (int*)malloc(nbytes);
    h_B = (int*)malloc(nbytes);
    for (int i = 0; i < n; i++) {
        h_A[i] = A[i]; h_B[i] = B[i];
    }

    int* d_A, * d_B, *d_odata;
    int blocksize = 512;
    int size = n;
    dim3 block(blocksize, 1);
    dim3 grid((nbytes + blocksize - 1) / blocksize, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes));
    CHECK(cudaMalloc((void**)&d_B, nbytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), gird(%d), block(%d)\n", nbytes, grid.x, block.x);
    compareArrayElement << <grid, block >> > (d_A, d_B, d_odata, n);
    int* h_odata = (int*)malloc(grid.x * sizeof(int));
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    int d_answer = 0;
    for (int i = 0; i < grid.x; i++) d_answer += h_odata[i];
    if (d_answer > 0) d_answer = 1;
    else d_answer = 0;
    printf("[host] device answer : %d\n", d_answer);
    checkResultInt(&h_answer, &d_answer, 1);

    // memory free
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_odata);
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==24600== NVPROF is profiling process 24600, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 1
[host] datasize (40000), gird(79), block(512)
[host] device answer : 1
[host] Arrays match.

==24600== Profiling application: ./Cuda.exe 1
==24600== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24600== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   48.70%  13.697us         2  6.8480us  6.3360us  7.3610us  [CUDA memcpy HtoD]
                   43.23%  12.159us         1  12.159us  12.159us  12.159us  compareArrayElement(int*, int*, int*, unsigned int)
                    8.08%  2.2720us         1  2.2720us  2.2720us  2.2720us  [CUDA memcpy DtoH]
      API calls:   68.56%  67.964ms         1  67.964ms  67.964ms  67.964ms  cudaSetDevice
                   29.31%  29.056ms         1  29.056ms  29.056ms  29.056ms  cuDevicePrimaryCtxRelease
                    1.21%  1.1946ms         1  1.1946ms  1.1946ms  1.1946ms  cudaLaunchKernel
                    0.43%  429.40us         3  143.13us  61.400us  189.70us  cudaMemcpy
                    0.27%  272.50us         3  90.833us  4.0000us  260.70us  cudaMalloc
                    0.10%  103.00us         1  103.00us  103.00us  103.00us  cuLibraryUnload
                    0.06%  56.200us       114     492ns       0ns  21.700us  cuDeviceGetAttribute
                    0.04%  40.800us         2  20.400us  4.7000us  36.100us  cudaFree
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cudaGetDeviceProperties
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         3     600ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==23592== NVPROF is profiling process 23592, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 0
[host] datasize (40000), gird(79), block(512)
[host] device answer : 0
[host] Arrays match.

==23592== Profiling application: ./Cuda.exe 2
==23592== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==23592== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.22%  27.360us         2  13.680us  6.1120us  21.248us  [CUDA memcpy HtoD]
                   29.06%  12.192us         1  12.192us  12.192us  12.192us  compareArrayElement(int*, int*, int*, unsigned int)
                    5.72%  2.3990us         1  2.3990us  2.3990us  2.3990us  [CUDA memcpy DtoH]
      API calls:   69.97%  67.523ms         1  67.523ms  67.523ms  67.523ms  cudaSetDevice
                   27.93%  26.950ms         1  26.950ms  26.950ms  26.950ms  cuDevicePrimaryCtxRelease
                    1.03%  994.40us         1  994.40us  994.40us  994.40us  cudaLaunchKernel
                    0.62%  599.80us         3  199.93us  85.200us  409.40us  cudaMemcpy
                    0.32%  306.20us         3  102.07us  2.5000us  297.20us  cudaMalloc
                    0.06%  54.100us         1  54.100us  54.100us  54.100us  cuLibraryUnload
                    0.04%  35.500us         2  17.750us  15.900us  19.600us  cudaFree
                    0.02%  18.800us       114     164ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.02%  17.700us         1  17.700us  17.700us  17.700us  cudaGetDeviceProperties
                    0.00%  2.4000us         3     800ns     100ns  2.0000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==8732== NVPROF is profiling process 8732, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 1
[host] datasize (40000), gird(79), block(512)
[host] device answer : 1
[host] Arrays match.

==8732== Profiling application: ./Cuda.exe 3
==8732== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==8732== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   45.93%  12.288us         1  12.288us  12.288us  12.288us  compareArrayElement(int*, int*, int*, unsigned int)
                   45.10%  12.064us         2  6.0320us  5.9520us  6.1120us  [CUDA memcpy HtoD]
                    8.97%  2.4000us         1  2.4000us  2.4000us  2.4000us  [CUDA memcpy DtoH]
      API calls:   67.93%  68.124ms         1  68.124ms  68.124ms  68.124ms  cudaSetDevice
                   29.75%  29.838ms         1  29.838ms  29.838ms  29.838ms  cuDevicePrimaryCtxRelease
                    1.04%  1.0422ms         1  1.0422ms  1.0422ms  1.0422ms  cudaLaunchKernel
                    0.79%  792.60us         3  264.20us  69.600us  518.90us  cudaMemcpy
                    0.30%  297.80us         3  99.266us  4.4000us  287.90us  cudaMalloc
                    0.11%  113.00us         1  113.00us  113.00us  113.00us  cuLibraryUnload
                    0.04%  44.300us         2  22.150us  6.3000us  38.000us  cudaFree
                    0.02%  20.100us       114     176ns       0ns  3.2000us  cuDeviceGetAttribute
                    0.00%  3.2000us         1  3.2000us  3.2000us  3.2000us  cudaGetDeviceProperties
                    0.00%  1.8000us         3     600ns       0ns  1.5000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 4
==21936== NVPROF is profiling process 21936, command: ./Cuda.exe 4
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 0
[host] datasize (8), gird(1), block(512)
[host] device answer : 0
[host] Arrays match.

==21936== Profiling application: ./Cuda.exe 4
==21936== Warning: 16 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21936== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.82%  8.7360us         1  8.7360us  8.7360us  8.7360us  compareArrayElement(int*, int*, int*, unsigned int)
                   19.44%  2.4320us         1  2.4320us  2.4320us  2.4320us  [CUDA memcpy DtoH]
                   10.74%  1.3440us         2     672ns     352ns     992ns  [CUDA memcpy HtoD]
      API calls:   66.86%  70.417ms         1  70.417ms  70.417ms  70.417ms  cudaSetDevice
                   31.02%  32.673ms         1  32.673ms  32.673ms  32.673ms  cuDevicePrimaryCtxRelease
                    0.97%  1.0227ms         1  1.0227ms  1.0227ms  1.0227ms  cudaLaunchKernel
                    0.62%  652.80us         3  217.60us  81.800us  413.40us  cudaMemcpy
                    0.26%  272.50us         3  90.833us  2.4000us  264.90us  cudaMalloc
                    0.12%  129.90us         1  129.90us  129.90us  129.90us  cuLibraryUnload
                    0.06%  61.900us       114     542ns       0ns  15.500us  cuDeviceGetAttribute
                    0.06%  61.300us         2  30.650us  6.3000us  55.000us  cudaFree
                    0.02%  23.600us         1  23.600us  23.600us  23.600us  cuDeviceTotalMem
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuModuleGetLoadingMode
                    0.00%  2.6000us         3     866ns       0ns  2.3000us  cuDeviceGetCount
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
