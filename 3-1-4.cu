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
            printf("[host] host %5lld gpu %5lld at current %d\n", h_data[i], d_data[i], i);
            break;
        }
    }
    if (match) printf("[host] Arrays match.\n\n");
}

#define DIM 512
__global__ void addRange(int* d_A, const int n, int s, int e, int k) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (s <= idx && idx <= e) {
        d_A[idx] += k;
    }
}

__global__ void sumRange(int* d_A, long long* d_B, const int n, int s, int e) {
    __shared__ long long smem[DIM];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    if (s <= idx && idx <= e) smem[tid] = (long long)d_A[idx];
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
    vector<int>A(n), B;
    for (auto& a : A) {
        cin >> a; B.push_back(a);
    }
    vector<tuple<int, int, int, int>> C;
    vector<long long> D;
    for (int i = 0; i < m; i++) {
        int op; cin >> op;
        if (op == 1) {
            int x, y, z; cin >> x >> y >> z;
            for (int j = x; j <= y; j++)
                A[j] += z;
            C.emplace_back(op, x, y, z);
        }
        else {
            long long answer = 0;
            int x, y; cin >> x >> y;
            for (int j = x; j <= y; j++)
                answer += A[j];
            C.emplace_back(op, x, y, -1);
            //cout << answer << '\n';
            D.push_back(answer);
        }
    }
    int k = (int)D.size();
    long long* h_answer = (long long*)malloc(k * sizeof(long long));
    for (int i = 0; i < k; i++) {
        h_answer[i] = D[i];
    }
    //printf("[host] host answer : ");
    //for (int i = 0; i < k; i++) printf("%lld ", h_answer[i]);
    //printf("\n");

    /* host - device code */
    int nbytes = n * sizeof(int);
    int kbytes = k * sizeof(long long);
    int* h_A = (int*)malloc(nbytes);
    for (int i = 0; i < n; i++) h_A[i] = B[i];

    /* device code */
    int* d_A; long long* d_B;
    dim3 block(128, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes));
    CHECK(cudaMalloc((void**)&d_B, grid.x*sizeof(long long)));
    CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    long long* d_answer = (long long*)malloc(kbytes);
    int d_answer_idx = 0;
    long long* tmp = (long long*)malloc(grid.x * sizeof(long long));
    for (auto &c: C) {
        int op = get<0>(c), x = get<1>(c), y = get<2>(c), z = get<3>(c);
        if (op == 1) {
            CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
            addRange << <grid, block >> > (d_A, n, x, y, z);
            cudaDeviceSynchronize();
            CHECK(cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost));
        }
        else {
            sumRange << <grid, block >> > (d_A, d_B, n, x, y);
            cudaDeviceSynchronize();
            CHECK(cudaMemcpy(tmp, d_B, grid.x * sizeof(long long), cudaMemcpyDeviceToHost));
            long long x = 0;
            for (int i = 0; i < grid.x; i++) x += tmp[i];
            d_answer[d_answer_idx++] = x;
        }
    }
    //printf("[host] device answer : ");
    //for (int i = 0; i < k; i++) printf("%lld ", d_answer[i]);
    //printf("\n");
    checkResult<long long>(h_answer, d_answer, k);
    free(h_answer); free(h_A); free(d_answer); free(tmp);
    cudaFree(d_A); cudaFree(d_B);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==14072== NVPROF is profiling process 14072, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (40000), grid(79, 1), block(128, 1)
[host] Arrays match.

==14072== Profiling application: ./Cuda.exe 1
==14072== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==14072== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   41.75%  61.720ms     10000  6.1710us  1.1840us  25.311us  [CUDA memcpy DtoH]
                   29.90%  44.199ms      4970  8.8930us  7.5200us  10.752us  sumRange(int*, __int64*, int, int, int)
                   21.68%  32.042ms      5031  6.3680us  5.7280us  22.144us  [CUDA memcpy HtoD]
                    6.67%  9.8559ms      5030  1.9590us  1.5360us  2.4640us  addRange(int*, int, int, int, int)
      API calls:   41.80%  305.25ms     15031  20.308us  6.4000us  297.20us  cudaMemcpy
                   23.06%  168.40ms     10000  16.839us  2.3000us  172.80us  cudaDeviceSynchronize
                   19.13%  139.73ms     10000  13.972us  5.7000us  30.428ms  cudaLaunchKernel
                   12.33%  90.082ms         1  90.082ms  90.082ms  90.082ms  cudaSetDevice
                    3.60%  26.283ms         1  26.283ms  26.283ms  26.283ms  cuDevicePrimaryCtxRelease
                    0.04%  299.80us         2  149.90us  22.600us  277.20us  cudaFree
                    0.03%  226.10us         2  113.05us  10.200us  215.90us  cudaMalloc
                    0.01%  46.700us         1  46.700us  46.700us  46.700us  cuLibraryUnload
                    0.00%  19.300us       114     169ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  3.6000us         1  3.6000us  3.6000us  3.6000us  cudaGetDeviceProperties
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuModuleGetLoadingMode
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==35472== NVPROF is profiling process 35472, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (40000), grid(79, 1), block(128, 1)
[host] Arrays match.

==35472== Profiling application: ./Cuda.exe 2
==35472== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==35472== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.52%  61.108ms     10000  6.1100us  1.2150us  25.024us  [CUDA memcpy DtoH]
                   31.68%  47.767ms      5063  9.4340us  8.1920us  10.528us  sumRange(int*, __int64*, int, int, int)
                   21.03%  31.713ms      4938  6.4220us  5.7590us  22.592us  [CUDA memcpy HtoD]
                    6.77%  10.211ms      4937  2.0680us  1.7270us  2.8480us  addRange(int*, int, int, int, int)
      API calls:   44.53%  332.52ms     14938  22.259us  6.5000us  334.40us  cudaMemcpy
                   25.86%  193.08ms     10000  19.307us  2.9000us  152.60us  cudaDeviceSynchronize
                   14.97%  111.78ms     10000  11.177us  5.7000us  1.3402ms  cudaLaunchKernel
                   10.74%  80.232ms         1  80.232ms  80.232ms  80.232ms  cudaSetDevice
                    3.80%  28.377ms         1  28.377ms  28.377ms  28.377ms  cuDevicePrimaryCtxRelease
                    0.05%  384.10us         2  192.05us  43.400us  340.70us  cudaFree
                    0.03%  218.20us         2  109.10us  5.7000us  212.50us  cudaMalloc
                    0.01%  57.400us         1  57.400us  57.400us  57.400us  cuLibraryUnload
                    0.01%  55.800us       114     489ns       0ns  29.400us  cuDeviceGetAttribute
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.3000us         3     766ns     100ns  2.0000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==30560== NVPROF is profiling process 30560, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (40000), grid(79, 1), block(128, 1)
[host] Arrays match.

==30560== Profiling application: ./Cuda.exe 3
==30560== Warning: 27 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==30560== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   41.29%  61.652ms     10000  6.1650us  1.1840us  25.503us  [CUDA memcpy DtoH]
                   30.54%  45.606ms      4981  9.1560us  7.5840us  12.927us  sumRange(int*, __int64*, int, int, int)
                   21.38%  31.922ms      5020  6.3590us  5.6960us  31.616us  [CUDA memcpy HtoD]
                    6.79%  10.142ms      5019  2.0200us  1.5990us  3.4560us  addRange(int*, int, int, int, int)
      API calls:   43.76%  310.39ms     15020  20.665us  6.6000us  251.50us  cudaMemcpy
                   23.59%  167.36ms     10000  16.735us  2.5000us  102.70us  cudaDeviceSynchronize
                   16.55%  117.42ms     10000  11.741us  5.7000us  1.3745ms  cudaLaunchKernel
                   12.61%  89.452ms         1  89.452ms  89.452ms  89.452ms  cudaSetDevice
                    3.41%  24.207ms         1  24.207ms  24.207ms  24.207ms  cuDevicePrimaryCtxRelease
                    0.03%  233.80us         2  116.90us  19.600us  214.20us  cudaFree
                    0.03%  232.90us         2  116.45us  4.8000us  228.10us  cudaMalloc
                    0.01%  38.200us         1  38.200us  38.200us  38.200us  cuLibraryUnload
                    0.00%  20.000us       114     175ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns       0ns  1.8000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
