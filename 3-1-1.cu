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

#define DIM 512
__global__ void countGreaterOrEqual(long long* d_A, int* d_B, const int n, long long k) {
    __shared__ int smem[DIM];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // boudary check
    if (idx >= n) return;

    if (d_A[idx] >= k) smem[tid] = 1;
    else smem[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && idx + stride < n) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) d_B[blockIdx.x] = smem[0];
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

    int n, m; cin >> n >> m;
    vector<long long> A(n);
    for (auto& a : A) cin >> a;
    int* h_answer = (int*)malloc(m * sizeof(int));
    memset(h_answer, 0, m * sizeof(int));
    long long* B = (long long *)malloc(m * sizeof(long long));
    for (int i = 0; i < m; i++) {
        cin >> B[i];
        for (int j = 0; j < n; j++) {
            if (A[j] >= B[i]) h_answer[i]++;
        }
    }
    //printf("[host] host answer : ");
    //for (int i = 0; i < m; i++) printf("%d ", h_answer[i]);
    //printf("\n");

    /* host - device code */
    int nbytes = n * sizeof(long long);
    int mbytes = m * sizeof(int);
    long long* h_A = (long long*)malloc(nbytes);
    int* h_B = (int*)malloc(mbytes);
    for (int i = 0; i < n; i++) h_A[i] = A[i];

    /* device code */
    long long* d_A; int* d_B;
    dim3 block(128, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes));
    CHECK(cudaMalloc((void**)&d_B, grid.x * sizeof(int)));
    CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    int* d_answer = (int*)malloc(mbytes);
    memset(d_answer, 0, mbytes);
    int* tmp = (int*)malloc(grid.x * sizeof(int));
    for (int i = 0; i < m; i++) {
        countGreaterOrEqual << <grid, block >> > (d_A, d_B, n, B[i]);
        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(tmp, d_B, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int j = 0; j < grid.x; j++) {
            d_answer[i] += tmp[j];
        }
    }
    //printf("[host] device answer : ");
    //for (int i = 0; i < m; i++) printf("%d ", d_answer[i]);
    //printf("\n");
    checkResult<int>(h_answer, d_answer, m);
    free(h_answer); free(d_answer); free(tmp);
    cudaFree(d_A); cudaFree(d_B);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==35408== NVPROF is profiling process 35408, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (8000), grid(8, 1), block(128, 1)
[host] Arrays match.

==35408== Profiling application: ./Cuda.exe 1
==35408== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==35408== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   87.21%  9.8250ms      1000  9.8240us  9.5360us  10.656us  countGreaterOrEqual(__int64*, int*, int, __int64)
                   12.76%  1.4377ms      1000  1.4370us  1.2800us  2.1440us  [CUDA memcpy DtoH]
                    0.02%  2.6240us         1  2.6240us  2.6240us  2.6240us  [CUDA memcpy HtoD]
      API calls:   51.36%  84.459ms         1  84.459ms  84.459ms  84.459ms  cudaSetDevice
                   14.59%  23.990ms         1  23.990ms  23.990ms  23.990ms  cuDevicePrimaryCtxRelease
                   12.00%  19.740ms      1000  19.740us  13.400us  62.700us  cudaDeviceSynchronize
                   11.07%  18.198ms      1000  18.198us  6.6000us  1.3791ms  cudaLaunchKernel
                   10.67%  17.549ms      1001  17.531us  11.700us  102.40us  cudaMemcpy
                    0.14%  229.50us         2  114.75us  7.8000us  221.70us  cudaMalloc
                    0.13%  221.50us         2  110.75us  24.500us  197.00us  cudaFree
                    0.02%  34.400us         1  34.400us  34.400us  34.400us  cuLibraryUnload
                    0.01%  18.800us       114     164ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  6.4000us         1  6.4000us  6.4000us  6.4000us  cudaGetDeviceProperties
                    0.00%  2.1000us         2  1.0500us     100ns  2.0000us  cuDeviceGet
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.7000us         3     566ns     100ns  1.4000us  cuDeviceGetCount
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==14120== NVPROF is profiling process 14120, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (8000), grid(8, 1), block(128, 1)
[host] Arrays match.

==14120== Profiling application: ./Cuda.exe 2
==14120== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==14120== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.48%  9.7723ms      1000  9.7720us  9.3760us  10.336us  countGreaterOrEqual(__int64*, int*, int, __int64)
                   13.51%  1.5262ms      1000  1.5260us  1.4080us  3.8720us  [CUDA memcpy DtoH]
                    0.02%  2.1760us         1  2.1760us  2.1760us  2.1760us  [CUDA memcpy HtoD]
      API calls:   35.24%  67.443ms         1  67.443ms  67.443ms  67.443ms  cudaSetDevice
                   26.75%  51.198ms      1000  51.197us  3.6000us  255.70us  cudaDeviceSynchronize
                   17.62%  33.715ms      1001  33.681us  14.400us  207.70us  cudaMemcpy
                   13.38%  25.610ms         1  25.610ms  25.610ms  25.610ms  cuDevicePrimaryCtxRelease
                    6.66%  12.746ms      1000  12.745us  9.3000us  1.0595ms  cudaLaunchKernel
                    0.17%  317.10us         2  158.55us  21.800us  295.30us  cudaFree
                    0.13%  253.20us         2  126.60us  6.3000us  246.90us  cudaMalloc
                    0.02%  33.400us         1  33.400us  33.400us  33.400us  cuLibraryUnload
                    0.02%  32.400us       114     284ns       0ns  15.100us  cuDeviceGetAttribute
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  3.7000us         3  1.2330us     100ns  3.4000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==27472== NVPROF is profiling process 27472, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (8000), grid(8, 1), block(128, 1)
[host] Arrays match.

==27472== Profiling application: ./Cuda.exe 3
==27472== Warning: 35 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==27472== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   86.57%  9.8126ms      1000  9.8120us  9.5350us  10.495us  countGreaterOrEqual(__int64*, int*, int, __int64)
                   13.41%  1.5203ms      1000  1.5200us  1.2800us  8.3520us  [CUDA memcpy DtoH]
                    0.02%  2.2080us         1  2.2080us  2.2080us  2.2080us  [CUDA memcpy HtoD]
      API calls:   47.02%  66.271ms         1  66.271ms  66.271ms  66.271ms  cudaSetDevice
                   18.05%  25.438ms         1  25.438ms  25.438ms  25.438ms  cuDevicePrimaryCtxRelease
                   13.58%  19.135ms      1001  19.115us  11.100us  326.00us  cudaMemcpy
                   12.57%  17.720ms      1000  17.720us  2.4000us  43.800us  cudaDeviceSynchronize
                    8.42%  11.866ms      1000  11.865us  5.7000us  1.1369ms  cudaLaunchKernel
                    0.20%  281.30us         2  140.65us  20.000us  261.30us  cudaFree
                    0.12%  167.50us         2  83.750us  5.1000us  162.40us  cudaMalloc
                    0.02%  30.000us         1  30.000us  30.000us  30.000us  cuLibraryUnload
                    0.01%  19.600us       114     171ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.00%  6.4000us         1  6.4000us  6.4000us  6.4000us  cudaGetDeviceProperties
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
