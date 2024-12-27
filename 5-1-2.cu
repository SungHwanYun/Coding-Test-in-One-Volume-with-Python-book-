#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<bits/stdc++.h>
#include<time.h>
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

    clock_t start = clock();
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
    int times = 0;
    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] host time : %d ms\n", times);
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
    start = clock();
    int* d_A; long long* d_B;
    dim3 block(128, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes));
    CHECK(cudaMalloc((void**)&d_B, grid.x * sizeof(long long)));
    CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    long long* d_answer = (long long*)malloc(kbytes);
    int d_answer_idx = 0;
    long long* tmp = (long long*)malloc(grid.x * sizeof(long long));
    for (auto& c : C) {
        int op = get<0>(c), x = get<1>(c), y = get<2>(c), z = get<3>(c);
        if (op == 1) {
            //CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
            addRange << <grid, block >> > (d_A, n, x, y, z);
            cudaDeviceSynchronize();
            //CHECK(cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost));
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
    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] device time : %d ms\n", times);
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
==24624== NVPROF is profiling process 24624, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 29597 ms
[host] datasize (400000), grid(782, 1), block(128, 1)
[host] device time : 2657 ms
[host] Arrays match.

==24624== Profiling application: ./Cuda.exe 1
==24624== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24624== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.95%  1.04440s     99999  10.444us  8.6720us  147.20us  addRange(int*, int, int, int, int)
                    0.05%  510.78us         1  510.78us  510.78us  510.78us  [CUDA memcpy HtoD]
                    0.00%  50.527us         1  50.527us  50.527us  50.527us  sumRange(int*, __int64*, int, int, int)
                    0.00%  1.5680us         1  1.5680us  1.5680us  1.5680us  [CUDA memcpy DtoH]
      API calls:   62.57%  1.69575s    100000  16.957us  2.2000us  462.60us  cudaDeviceSynchronize
                   32.70%  886.17ms    100000  8.8610us  5.6000us  2.3133ms  cudaLaunchKernel
                    3.65%  99.059ms         1  99.059ms  99.059ms  99.059ms  cudaSetDevice
                    1.06%  28.656ms         1  28.656ms  28.656ms  28.656ms  cuDevicePrimaryCtxRelease
                    0.01%  254.70us         2  127.35us  5.0000us  249.70us  cudaMalloc
                    0.01%  208.20us         2  104.10us  85.700us  122.50us  cudaMemcpy
                    0.01%  196.10us         2  98.050us  19.200us  176.90us  cudaFree
                    0.00%  40.700us         1  40.700us  40.700us  40.700us  cuLibraryUnload
                    0.00%  19.800us       114     173ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cuModuleGetLoadingMode
                    0.00%  2.2000us         3     733ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
