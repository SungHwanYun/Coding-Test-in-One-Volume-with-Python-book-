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

__device__ int get_num(char* s) {
    int h = (s[0] - '0') * 10 + (s[1] - '0');
    int m = (s[3] - '0') * 10 + (s[4] - '0');
    return h * 60 + m;
}
#define DIM 32
__global__ void sumStringNum(char* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = t_idx * 5;
    __shared__ int smem[DIM];

    // boundary check
    if (t_idx >= n) return;
    smem[tid] = get_num(g_idata + idx);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = smem[tid];
}

int parse_log(string& s) {
    string hour = s.substr(0, 2);
    string minute = s.substr(3, 2);
    return stoi(hour) * 60 + stoi(minute);
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

    vector<string> A;
    do {
        string a; cin >> a;
        A.push_back(a);
    } while (getc(stdin) == ' ');

    int total_time = 0;
    for (auto& a : A) {
        int t = parse_log(a);
        total_time += t;
    }

    int hour = total_time / 60;
    int minute = total_time % 60;

    char buf[100];
    if (hour < 100) sprintf(buf, "%02d:%02d", hour, minute);
    else sprintf(buf, "%d:%02d", hour, minute);
    char* h_answer = (char *)malloc(strlen(buf) * sizeof(char) + 1);
    strcpy(h_answer, buf);
    printf("[host] host answer : %s\n", h_answer);

    /* host - device code */
    int n = (int)A.size();
    int nbytes = n * 5 * sizeof(char);
    char* h_A = (char*)malloc(nbytes);
    for (int i = 0; i < n; i++) {
        memcpy(h_A + i * 5, A[i].c_str(), 5);
    }

    /* device code */
    char* d_A; int* d_odata;
    dim3 block(32, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    sumStringNum << <grid, block >> > (d_A, d_odata, n);
    int* d_answer2 = (int*)malloc(grid.x*sizeof(int));
    cudaMemcpy(d_answer2, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    total_time = 0;
    for (int i = 0; i < grid.x; i++) total_time += d_answer2[i];
    hour = total_time / 60;
    minute = total_time % 60;
    if (hour < 100) sprintf(buf, "%02d:%02d", hour, minute);
    else sprintf(buf, "%d:%02d", hour, minute);
    char* d_answer = (char*)malloc(strlen(buf) * sizeof(char) + 1);
    strcpy(d_answer, buf);
    printf("[host] device answer : %s\n", d_answer);
    checkResult<char>(h_answer, d_answer, strlen(buf) * sizeof(char));

    // memory free
    free(h_A); free(h_answer); free(d_answer);
    cudaFree(d_A); cudaFree(d_odata);
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==28160== NVPROF is profiling process 28160, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 121025:41
[host] datasize (50000), grid(313, 1), block(32, 1)
[host] device answer : 121025:41
[host] Arrays match.

==28160== Profiling application: ./Cuda.exe 1
==28160== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28160== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.16%  15.424us         1  15.424us  15.424us  15.424us  sumStringNum(char*, int*, unsigned int)
                   39.94%  11.808us         1  11.808us  11.808us  11.808us  [CUDA memcpy HtoD]
                    7.90%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
      API calls:   67.89%  68.675ms         1  68.675ms  68.675ms  68.675ms  cudaSetDevice
                   29.42%  29.761ms         1  29.761ms  29.761ms  29.761ms  cuDevicePrimaryCtxRelease
                    2.02%  2.0413ms         1  2.0413ms  2.0413ms  2.0413ms  cudaLaunchKernel
                    0.27%  274.10us         2  137.05us  6.0000us  268.10us  cudaMalloc
                    0.17%  176.50us         2  88.250us  81.900us  94.600us  cudaMemcpy
                    0.15%  156.20us         2  78.100us  16.100us  140.10us  cudaFree
                    0.03%  33.300us         1  33.300us  33.300us  33.300us  cuLibraryUnload
                    0.02%  20.800us       114     182ns       0ns  3.8000us  cuDeviceGetAttribute
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cuModuleGetLoadingMode
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==24208== NVPROF is profiling process 24208, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 119539:03
[host] datasize (50000), grid(313, 1), block(32, 1)
[host] device answer : 119539:03
[host] Arrays match.

==24208== Profiling application: ./Cuda.exe 2
==24208== Warning: 36 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24208== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.26%  15.520us         1  15.520us  15.520us  15.520us  sumStringNum(char*, int*, unsigned int)
                   34.36%  9.3120us         1  9.3120us  9.3120us  9.3120us  [CUDA memcpy HtoD]
                    8.38%  2.2720us         1  2.2720us  2.2720us  2.2720us  [CUDA memcpy DtoH]
      API calls:   72.64%  69.598ms         1  69.598ms  69.598ms  69.598ms  cudaSetDevice
                   24.78%  23.743ms         1  23.743ms  23.743ms  23.743ms  cuDevicePrimaryCtxRelease
                    1.85%  1.7733ms         1  1.7733ms  1.7733ms  1.7733ms  cudaLaunchKernel
                    0.25%  241.30us         2  120.65us  7.2000us  234.10us  cudaMalloc
                    0.23%  219.90us         2  109.95us  13.300us  206.60us  cudaFree
                    0.18%  174.10us         2  87.050us  62.300us  111.80us  cudaMemcpy
                    0.03%  31.100us         1  31.100us  31.100us  31.100us  cuLibraryUnload
                    0.02%  20.200us       114     177ns       0ns  3.3000us  cuDeviceGetAttribute
                    0.00%  3.2000us         1  3.2000us  3.2000us  3.2000us  cudaGetDeviceProperties
                    0.00%  1.9000us         3     633ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==31984== NVPROF is profiling process 31984, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 120146:53
[host] datasize (50000), grid(313, 1), block(32, 1)
[host] device answer : 120146:53
[host] Arrays match.

==31984== Profiling application: ./Cuda.exe 3
==31984== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==31984== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.64%  15.328us         1  15.328us  15.328us  15.328us  sumStringNum(char*, int*, unsigned int)
                   33.92%  9.3440us         1  9.3440us  9.3440us  9.3440us  [CUDA memcpy HtoD]
                   10.45%  2.8790us         1  2.8790us  2.8790us  2.8790us  [CUDA memcpy DtoH]
      API calls:   72.95%  68.529ms         1  68.529ms  68.529ms  68.529ms  cudaSetDevice
                   24.29%  22.819ms         1  22.819ms  22.819ms  22.819ms  cuDevicePrimaryCtxRelease
                    1.54%  1.4439ms         1  1.4439ms  1.4439ms  1.4439ms  cudaLaunchKernel
                    0.52%  484.20us         2  242.10us  45.400us  438.80us  cudaFree
                    0.32%  304.80us         2  152.40us  91.000us  213.80us  cudaMemcpy
                    0.32%  303.50us         2  151.75us  9.4000us  294.10us  cudaMalloc
                    0.03%  27.800us         1  27.800us  27.800us  27.800us  cuLibraryUnload
                    0.02%  19.700us       114     172ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  3.5000us         3  1.1660us     100ns  2.9000us  cuDeviceGetCount
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cudaGetDeviceProperties
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.3000us         2     650ns       0ns  1.3000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 4
==17788== NVPROF is profiling process 17788, command: ./Cuda.exe 4
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 00:00
[host] datasize (10), grid(1, 1), block(32, 1)
[host] device answer : 00:00
[host] Arrays match.

==17788== Profiling application: ./Cuda.exe 4
==17788== Warning: 17 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==17788== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   71.92%  8.1920us         1  8.1920us  8.1920us  8.1920us  sumStringNum(char*, int*, unsigned int)
                   19.38%  2.2070us         1  2.2070us  2.2070us  2.2070us  [CUDA memcpy DtoH]
                    8.70%     991ns         1     991ns     991ns     991ns  [CUDA memcpy HtoD]
      API calls:   65.09%  69.653ms         1  69.653ms  69.653ms  69.653ms  cudaSetDevice
                   32.62%  34.910ms         1  34.910ms  34.910ms  34.910ms  cuDevicePrimaryCtxRelease
                    0.97%  1.0393ms         1  1.0393ms  1.0393ms  1.0393ms  cudaLaunchKernel
                    0.50%  538.90us         2  269.45us  48.400us  490.50us  cudaFree
                    0.38%  406.30us         2  203.15us  51.500us  354.80us  cudaMalloc
                    0.31%  335.70us         2  167.85us  108.90us  226.80us  cudaMemcpy
                    0.07%  78.600us         1  78.600us  78.600us  78.600us  cuLibraryUnload
                    0.03%  28.000us       114     245ns       0ns  3.4000us  cuDeviceGetAttribute
                    0.01%  5.5000us         1  5.5000us  5.5000us  5.5000us  cuDeviceGetName
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cuModuleGetLoadingMode
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
