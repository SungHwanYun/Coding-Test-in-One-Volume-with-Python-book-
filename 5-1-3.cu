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
==21944== NVPROF is profiling process 21944, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 26697 ms
[host] datasize (400000), grid(782, 1), block(128, 1)
[host] device time : 5994 ms
[host] Arrays match.

==21944== Profiling application: ./Cuda.exe 1
==21944== Warning: 37 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21944== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   78.32%  2.52489s     50000  50.497us  49.888us  53.216us  sumRange(int*, __int64*, int, int, int)
                   19.30%  622.25ms     50000  12.445us  8.6080us  148.70us  addRange(int*, int, int, int, int)
                    2.36%  76.133ms     50000  1.5220us  1.2800us  16.736us  [CUDA memcpy DtoH]
                    0.02%  513.82us         1  513.82us  513.82us  513.82us  [CUDA memcpy HtoD]
      API calls:   65.44%  3.88403s    100000  38.840us  2.5000us  476.20us  cudaDeviceSynchronize
                   18.64%  1.10614s    100000  11.061us  5.5000us  1.7924ms  cudaLaunchKernel
                   14.15%  839.86ms     50001  16.796us  13.800us  463.10us  cudaMemcpy
                    1.28%  76.048ms         1  76.048ms  76.048ms  76.048ms  cudaSetDevice
                    0.48%  28.221ms         1  28.221ms  28.221ms  28.221ms  cuDevicePrimaryCtxRelease
                    0.01%  314.60us         2  157.30us  18.300us  296.30us  cudaFree
                    0.00%  265.50us         2  132.75us  8.4000us  257.10us  cudaMalloc
                    0.00%  53.900us         1  53.900us  53.900us  53.900us  cuLibraryUnload
                    0.00%  18.500us       114     162ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  4.2000us         1  4.2000us  4.2000us  4.2000us  cudaGetDeviceProperties
                    0.00%  2.5000us         3     833ns     100ns  2.1000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
