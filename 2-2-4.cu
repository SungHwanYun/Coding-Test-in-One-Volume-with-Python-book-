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

__global__ void convertSpecialUpperToLowerCase(char* g_idata, char* g_odata, int state, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    char c = g_idata[idx];
    if ('A' <= c && c <= 'Z') {
        int a = c - 'A';
        if (state & (1 << a)) {
            c = 'a' + a;
        }
    }
    g_odata[idx] = c;
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

    string S; cin >> S;
    string A = S;
    vector<char> B;
    do {
        char b; cin >> b;
        B.push_back(b);
    } while (getc(stdin) == ' ');

    for (int i = 0; i < B.size(); i++) {
        char b = B[i];
        for (int j = 0; j < A.size(); j++) {
            if (A[j] == b) A[j] = b + 'a' - 'A';
        }
    }
    int n = (int)A.length();
    int nbytes = n * sizeof(char);
    char* h_answer = (char*)malloc(nbytes + 1);
    strcpy(h_answer, A.c_str());
    A = S;
   // printf("[host] host answer : %s\n", h_answer);

    /* host - device code */
    char* h_A = (char*)malloc(nbytes + 1);
    int h_B = 0;
    strcpy(h_A, A.c_str());
    for (auto& b : B) {
        int x = b - 'A';
        h_B |= (1 << x);
    }

    /* device code */
    char* d_A, *d_odata;
    dim3 block(32, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes + 1));
    CHECK(cudaMalloc((void**)&d_odata, nbytes + 1));
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    convertSpecialUpperToLowerCase << <grid, block >> > (d_A, d_odata, h_B, n);
    char* d_answer = (char*)malloc(nbytes + 1);
    cudaMemcpy(d_answer, d_odata, nbytes, cudaMemcpyDeviceToHost);
    d_answer[n] = 0;
   // printf("[host] device answer : %s\n", d_answer);
    checkResult<char>(h_answer, d_answer, n);

    // memory free
    free(h_A); free(h_answer); free(d_answer);
    cudaFree(d_A); cudaFree(d_odata);
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==21084== NVPROF is profiling process 21084, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (100000), grid(3125, 1), block(32, 1)
[host] Arrays match.

==21084== Profiling application: ./Cuda.exe 1
==21084== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21084== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   35.35%  35.199us         1  35.199us  35.199us  35.199us  convertSpecialUpperToLowerCase(char*, char*, int, unsigned int)
                   35.03%  34.880us         1  34.880us  34.880us  34.880us  [CUDA memcpy HtoD]
                   29.63%  29.504us         1  29.504us  29.504us  29.504us  [CUDA memcpy DtoH]
      API calls:   73.72%  93.707ms         1  93.707ms  93.707ms  93.707ms  cudaSetDevice
                   24.41%  31.021ms         1  31.021ms  31.021ms  31.021ms  cuDevicePrimaryCtxRelease
                    0.95%  1.2075ms         1  1.2075ms  1.2075ms  1.2075ms  cudaLaunchKernel
                    0.45%  573.40us         2  286.70us  88.100us  485.30us  cudaFree
                    0.22%  276.70us         2  138.35us  86.400us  190.30us  cudaMemcpy
                    0.19%  239.90us         2  119.95us  5.4000us  234.50us  cudaMalloc
                    0.04%  49.800us         1  49.800us  49.800us  49.800us  cuLibraryUnload
                    0.02%  19.900us       114     174ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.3000us         3     766ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==25328== NVPROF is profiling process 25328, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (100000), grid(3125, 1), block(32, 1)
[host] Arrays match.

==25328== Profiling application: ./Cuda.exe 2
==25328== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==25328== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   35.67%  35.935us         1  35.935us  35.935us  35.935us  [CUDA memcpy HtoD]
                   34.97%  35.232us         1  35.232us  35.232us  35.232us  convertSpecialUpperToLowerCase(char*, char*, int, unsigned int)
                   29.35%  29.568us         1  29.568us  29.568us  29.568us  [CUDA memcpy DtoH]
      API calls:   70.36%  75.154ms         1  75.154ms  75.154ms  75.154ms  cudaSetDevice
                   27.78%  29.674ms         1  29.674ms  29.674ms  29.674ms  cuDevicePrimaryCtxRelease
                    1.09%  1.1656ms         1  1.1656ms  1.1656ms  1.1656ms  cudaLaunchKernel
                    0.32%  346.10us         2  173.05us  25.800us  320.30us  cudaFree
                    0.21%  219.90us         2  109.95us  49.900us  170.00us  cudaMemcpy
                    0.16%  169.80us         2  84.900us  32.300us  137.50us  cudaMalloc
                    0.05%  53.400us       114     468ns       0ns  34.800us  cuDeviceGetAttribute
                    0.02%  25.200us         1  25.200us  25.200us  25.200us  cuLibraryUnload
                    0.00%  4.6000us         1  4.6000us  4.6000us  4.6000us  cudaGetDeviceProperties
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuModuleGetLoadingMode
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==26452== NVPROF is profiling process 26452, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (100000), grid(3125, 1), block(32, 1)
[host] Arrays match.

==26452== Profiling application: ./Cuda.exe 3
==26452== Warning: 35 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==26452== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.75%  52.671us         1  52.671us  52.671us  52.671us  [CUDA memcpy HtoD]
                   29.99%  35.296us         1  35.296us  35.296us  35.296us  convertSpecialUpperToLowerCase(char*, char*, int, unsigned int)
                   25.26%  29.728us         1  29.728us  29.728us  29.728us  [CUDA memcpy DtoH]
      API calls:   70.84%  69.783ms         1  69.783ms  69.783ms  69.783ms  cudaSetDevice
                   26.91%  26.509ms         1  26.509ms  26.509ms  26.509ms  cuDevicePrimaryCtxRelease
                    1.16%  1.1459ms         1  1.1459ms  1.1459ms  1.1459ms  cudaLaunchKernel
                    0.50%  493.50us         2  246.75us  40.200us  453.30us  cudaFree
                    0.26%  256.10us         2  128.05us  96.400us  159.70us  cudaMemcpy
                    0.21%  204.30us         2  102.15us  5.3000us  199.00us  cudaMalloc
                    0.08%  81.500us         1  81.500us  81.500us  81.500us  cuLibraryUnload
                    0.02%  22.800us       114     200ns       0ns  4.2000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
