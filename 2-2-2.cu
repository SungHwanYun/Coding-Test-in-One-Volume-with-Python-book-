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
template<typename T>
void checkResult(T* h_data, T *d_data, const int n) {
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

#define DIM 1024
__global__ void Kogge_Stone_scan_kernel(char* d_A, int* d_B, unsigned int n) {
    __shared__ int AB[DIM];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    if ('a' <= d_A[idx] && d_A[idx] <= 'z') {
        AB[threadIdx.x] = 1;
    }
    else {
        AB[threadIdx.x] = 0;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride*=2) {
        __syncthreads();
        int temp;
        if (threadIdx.x >= stride)
            temp = AB[threadIdx.x] + AB[threadIdx.x - stride];
        __syncthreads();
        if (threadIdx.x >= stride) {
            AB[threadIdx.x] = temp;
        }
    }
    if (idx < n) {
        d_B[idx] = AB[threadIdx.x];
    }
}
__global__ void removeUpperCase(char* d_A, char* d_A_out, int* d_B, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    if ('a' <= d_A[idx] && d_A[idx] <= 'z') {
        d_A_out[d_B[idx] - 1] = d_A[idx];
    }
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

    string A; cin >> A;
    string B;
    for (int i = 0; i < A.size(); i++) {
        if ('a' <= A[i] && A[i] <= 'z') B = B + A[i];
    }
    int m = (int)B.length();
    char* h_answer = (char*)malloc((m + 1) * sizeof(char));
    strcpy(h_answer, B.c_str());
    //printf("[host] host answer : %s\n", h_answer);

    /* host - device code */
    int n = (int)A.size();
    int nbytes = n * sizeof(char);
    char* h_A = (char*)malloc(nbytes + 1);
    strcpy(h_A, A.c_str());

    /* device code */
    char* d_A; char* d_A_out;
    int* d_B;
    dim3 block(1024, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes + 1));
    CHECK(cudaMalloc((void**)&d_A_out, nbytes + 1));
    CHECK(cudaMemset(d_A_out, 0, nbytes + 1));
    CHECK(cudaMalloc((void**)&d_B, n * sizeof(int)));
    CHECK(cudaMemset(d_B, 0, n * sizeof(int)));
    cudaMemcpy(d_A, h_A, nbytes + 1, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    Kogge_Stone_scan_kernel << <grid, block >> > (d_A, d_B, n);
    cudaDeviceSynchronize();
    removeUpperCase << <grid, block >> > (d_A, d_A_out, d_B, n);
    cudaDeviceSynchronize();
    char* d_answer = (char*)malloc(m + 1);
    memset(d_answer, 0, m + 1);
    CHECK(cudaMemcpy(d_answer, d_A_out, m, cudaMemcpyDeviceToHost));
    //printf("[host] device answer : %s\n", d_answer);
    checkResult<char>(h_answer, d_answer, m);

    // memory free
    free(h_A); free(h_answer);
    cudaFree(d_A); cudaFree(d_A_out); cudaFree(d_B); cudaFree(d_answer);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==28984== NVPROF is profiling process 28984, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (1000), grid(1, 1), block(1024, 1)
[host] Arrays match.

==28984== Profiling application: ./Cuda.exe 1
==28984== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28984== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.05%  18.624us         1  18.624us  18.624us  18.624us  Kogge_Stone_scan_kernel(char*, int*, unsigned int)
                   11.98%  3.3280us         1  3.3280us  3.3280us  3.3280us  removeUpperCase(char*, char*, int*, unsigned int)
                    8.76%  2.4330us         2  1.2160us     993ns  1.4400us  [CUDA memset]
                    7.72%  2.1440us         1  2.1440us  2.1440us  2.1440us  [CUDA memcpy DtoH]
                    4.50%  1.2490us         1  1.2490us  1.2490us  1.2490us  [CUDA memcpy HtoD]
      API calls:   74.60%  82.709ms         1  82.709ms  82.709ms  82.709ms  cudaSetDevice
                   22.84%  25.326ms         1  25.326ms  25.326ms  25.326ms  cuDevicePrimaryCtxRelease
                    1.87%  2.0735ms         2  1.0368ms  18.500us  2.0550ms  cudaLaunchKernel
                    0.27%  294.70us         4  73.675us  1.8000us  275.10us  cudaFree
                    0.15%  167.90us         2  83.950us  62.200us  105.70us  cudaMemcpy
                    0.11%  127.30us         3  42.433us  3.0000us  105.70us  cudaMalloc
                    0.05%  55.400us         2  27.700us  5.9000us  49.500us  cudaMemset
                    0.04%  44.700us         2  22.350us  8.0000us  36.700us  cudaDeviceSynchronize
                    0.03%  36.500us         1  36.500us  36.500us  36.500us  cuLibraryUnload
                    0.02%  20.800us       114     182ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.00%  4.3000us         1  4.3000us  4.3000us  4.3000us  cudaGetDeviceProperties
                    0.00%  2.5000us         3     833ns       0ns  2.2000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==19512== NVPROF is profiling process 19512, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (1000), grid(1, 1), block(1024, 1)
[host] Arrays match.

==19512== Profiling application: ./Cuda.exe 2
==19512== Warning: 12 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==19512== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.13%  18.559us         1  18.559us  18.559us  18.559us  Kogge_Stone_scan_kernel(char*, int*, unsigned int)
                   11.69%  3.2320us         1  3.2320us  3.2320us  3.2320us  removeUpperCase(char*, char*, int*, unsigned int)
                    8.92%  2.4660us         2  1.2330us  1.0250us  1.4410us  [CUDA memset]
                    7.87%  2.1750us         1  2.1750us  2.1750us  2.1750us  [CUDA memcpy DtoH]
                    4.40%  1.2160us         1  1.2160us  1.2160us  1.2160us  [CUDA memcpy HtoD]
      API calls:   71.07%  67.109ms         1  67.109ms  67.109ms  67.109ms  cudaSetDevice
                   26.28%  24.815ms         1  24.815ms  24.815ms  24.815ms  cuDevicePrimaryCtxRelease
                    1.31%  1.2376ms         2  618.80us  11.000us  1.2266ms  cudaLaunchKernel
                    0.59%  554.90us         4  138.73us  3.7000us  525.50us  cudaFree
                    0.29%  273.00us         2  136.50us  82.600us  190.40us  cudaMemcpy
                    0.17%  160.20us         3  53.400us  2.7000us  120.20us  cudaMalloc
                    0.12%  112.80us         1  112.80us  112.80us  112.80us  cuLibraryUnload
                    0.06%  60.900us         2  30.450us  5.6000us  55.300us  cudaMemset
                    0.06%  52.900us         2  26.450us  7.9000us  45.000us  cudaDeviceSynchronize
                    0.04%  36.800us       114     322ns       0ns  12.200us  cuDeviceGetAttribute
                    0.01%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         3     600ns     100ns  1.4000us  cuDeviceGetCount
                    0.00%  1.2000us         2     600ns       0ns  1.2000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==20284== NVPROF is profiling process 20284, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (1000), grid(1, 1), block(1024, 1)
[host] Arrays match.

==20284== Profiling application: ./Cuda.exe 3
==20284== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==20284== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   66.63%  18.720us         1  18.720us  18.720us  18.720us  Kogge_Stone_scan_kernel(char*, int*, unsigned int)
                   11.84%  3.3270us         1  3.3270us  3.3270us  3.3270us  removeUpperCase(char*, char*, int*, unsigned int)
                    8.77%  2.4640us         2  1.2320us  1.0240us  1.4400us  [CUDA memset]
                    7.75%  2.1760us         1  2.1760us  2.1760us  2.1760us  [CUDA memcpy DtoH]
                    5.01%  1.4080us         1  1.4080us  1.4080us  1.4080us  [CUDA memcpy HtoD]
      API calls:   71.47%  61.935ms         1  61.935ms  61.935ms  61.935ms  cudaSetDevice
                   26.28%  22.773ms         1  22.773ms  22.773ms  22.773ms  cuDevicePrimaryCtxRelease
                    1.16%  1.0089ms         2  504.45us  11.700us  997.20us  cudaLaunchKernel
                    0.39%  341.70us         4  85.425us  1.2000us  310.90us  cudaFree
                    0.20%  175.50us         2  87.750us  46.800us  128.70us  cudaMemcpy
                    0.17%  149.80us         1  149.80us  149.80us  149.80us  cuLibraryUnload
                    0.17%  149.50us         3  49.833us  2.7000us  142.50us  cudaMalloc
                    0.08%  69.700us         2  34.850us  32.700us  37.000us  cudaMemset
                    0.03%  28.300us         2  14.150us  8.3000us  20.000us  cudaDeviceSynchronize
                    0.02%  19.500us       114     171ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  4.2000us         1  4.2000us  4.2000us  4.2000us  cudaGetDeviceProperties
                    0.00%  2.8000us         3     933ns       0ns  2.3000us  cuDeviceGetCount
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
