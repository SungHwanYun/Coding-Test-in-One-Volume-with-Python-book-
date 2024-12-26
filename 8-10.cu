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
__global__ void addRange(int* d_A, const int nx, int ny, int sy, int sx, int ey, int ex, int k) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    // boundary check
    if (sx <= ix && ix <= ex && sy <= iy && iy <= ey) {
        d_A[idx] += k;
    }
}

__global__ void sumRange(int* d_A, long long* d_B, const int nx, int ny, int sy, int sx, int ey, int ex) {
    __shared__ long long smem[DIM];
    unsigned int tid = threadIdx.x;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (sx <= ix && ix <= ex && sy <= iy && iy <= ey) smem[tid] = d_A[idx];
    else smem[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && ix + stride < nx) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) d_B[blockIdx.y*gridDim.x + blockIdx.x] = smem[0];
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
    vector<vector<int>>A(n, vector<int>(n));
    int* h_A = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j]; h_A[i * n + j] = A[i][j];
        }
    }
    vector<tuple<int, int, int, int, int, int>> B;
    vector<long long> D;
    for (int i = 0; i < m; i++) {
        int op; cin >> op;
        int i1, j1, i2, j2, k;
        cin >> i1 >> j1 >> i2 >> j2; k = 0;
        if (op == 1) {
            cin >> k;
            for (int i = i1; i <= i2; i++) {
                for (int j = j1; j <= j2; j++) {
                    A[i][j] += k;
                }
            }
        }
        else {
            long long answer = 0;
            for (int i = i1; i <= i2; i++) {
                for (int j = j1; j <= j2; j++) {
                    answer += A[i][j];
                }
            }
            D.push_back(answer);
            //cout << answer << '\n';
        }
        B.emplace_back(op, i1, j1, i2, j2, k);
    }
    int k = (int)D.size();
    long long* h_answer = (long long*)malloc(k * sizeof(long long));
    for (int i = 0; i < k; i++) {
        h_answer[i] = D[i];
    }
   // printf("[host] host answer : ");
    //for (int i = 0; i < k; i++) printf("%lld ", h_answer[i]);
    //printf("\n");

    /* host - device code */
    int nbytes = n * n * sizeof(int);
    int kbytes = k * sizeof(long long);

    /* device code */
    int* d_A; long long* d_B;
    int nx = n, ny = n;
    dim3 block(128, 1);
    dim3 grid((nx + block.x - 1) / block.x, ny);
    CHECK(cudaMalloc((void**)&d_A, nbytes));
    CHECK(cudaMalloc((void**)&d_B, grid.x * grid.y * sizeof(long long)));
    CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    long long* d_answer = (long long*)malloc(kbytes);
    int d_answer_idx = 0;
    long long* tmp = (long long*)malloc(grid.x * grid.y * sizeof(long long));
    for (auto &b: B) {
        int op = get<0>(b), sy = get<1>(b), sx = get<2>(b), ey = get<3>(b), ex = get<4>(b), k = get<5>(b);
        if (op == 1) {
            CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
            addRange << <grid, block >> > (d_A, nx, ny, sy, sx, ey, ex, k);
            cudaDeviceSynchronize();
            CHECK(cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost));
        }
        else {
            sumRange << <grid, block >> > (d_A, d_B, nx, ny, sy, sx, ey, ex);
            cudaDeviceSynchronize();
            CHECK(cudaMemcpy(tmp, d_B, grid.x * grid.y * sizeof(long long), cudaMemcpyDeviceToHost));
            long long x = 0;
            for (int i = 0; i < grid.x; i++) for (int j = 0; j < grid.y; j++) x += tmp[i * grid.x + j];
            d_answer[d_answer_idx++] = x;
        }
    }
   // printf("[host] device answer : ");
   // for (int i = 0; i < k; i++) printf("%lld ", d_answer[i]);
  //  printf("\n");
    checkResult<long long>(h_answer, d_answer, k);
    free(h_answer); free(h_A); free(d_answer); free(tmp);
    cudaFree(d_A); cudaFree(d_B);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==18564== NVPROF is profiling process 18564, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (16), grid(1, 2), block(128, 1)
[host] Arrays match.

==18564== Profiling application: ./Cuda.exe 1
==18564== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==18564== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   37.71%  11.936us         4  2.9840us  2.8160us  3.4880us  addRange(int*, int, int, int, int, int, int, int)
                   28.61%  9.0560us         1  9.0560us  9.0560us  9.0560us  sumRange(int*, __int64*, int, int, int, int, int, int)
                   26.29%  8.3210us         5  1.6640us  1.4720us  2.2410us  [CUDA memcpy DtoH]
                    7.38%  2.3360us         5     467ns     320ns     992ns  [CUDA memcpy HtoD]
      API calls:   74.48%  78.260ms         1  78.260ms  78.260ms  78.260ms  cudaSetDevice
                   23.44%  24.629ms         1  24.629ms  24.629ms  24.629ms  cuDevicePrimaryCtxRelease
                    1.19%  1.2521ms         5  250.42us  8.3000us  1.2059ms  cudaLaunchKernel
                    0.30%  319.10us         2  159.55us  15.100us  304.00us  cudaFree
                    0.22%  231.40us        10  23.140us  5.4000us  78.700us  cudaMemcpy
                    0.15%  159.50us         2  79.750us  5.0000us  154.50us  cudaMalloc
                    0.11%  116.60us         1  116.60us  116.60us  116.60us  cuLibraryUnload
                    0.05%  55.200us         5  11.040us  8.5000us  14.300us  cudaDeviceSynchronize
                    0.04%  38.500us       114     337ns       0ns  21.300us  cuDeviceGetAttribute
                    0.00%  3.3000us         1  3.3000us  3.3000us  3.3000us  cudaGetDeviceProperties
                    0.00%  2.4000us         3     800ns       0ns  2.2000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.5000us         2     750ns       0ns  1.5000us  cuDeviceGet
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 12
==7844== NVPROF is profiling process 7844, command: ./Cuda.exe 12
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (315844), grid(3, 281), block(128, 1)
[host] Arrays do not match!
[host] host 174564104849 gpu     0 at current 0
==7844== Profiling application: ./Cuda.exe 12
==7844== Warning: 17 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==7844== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.71%  12.4500s    119971  103.78us  102.11us  289.15us  [CUDA memcpy HtoD]
                   45.33%  11.3525s    119971  94.626us  1.4720us  121.06us  [CUDA memcpy DtoH]
                    4.96%  1.24147s    119970  10.348us  9.5680us  25.184us  addRange(int*, int, int, int, int, int, int, int)
                    0.00%  57.376us         1  57.376us  57.376us  57.376us  sumRange(int*, __int64*, int, int, int, int, int, int)
      API calls:   51.96%  21.6746s    239942  90.332us  21.200us  7.6224ms  cudaMemcpy
                   44.33%  18.4908s    119971  154.13us  8.9000us  2.0259ms  cudaDeviceSynchronize
                    3.37%  1.40640s    119971  11.722us  6.7000us  1.3528ms  cudaLaunchKernel
                    0.27%  112.04ms         1  112.04ms  112.04ms  112.04ms  cudaSetDevice
                    0.07%  31.054ms         1  31.054ms  31.054ms  31.054ms  cuDevicePrimaryCtxRelease
                    0.00%  315.70us         2  157.85us  23.700us  292.00us  cudaFree
                    0.00%  273.30us         2  136.65us  5.5000us  267.80us  cudaMalloc
                    0.00%  63.700us         1  63.700us  63.700us  63.700us  cuLibraryUnload
                    0.00%  42.600us       114     373ns       0ns  17.800us  cuDeviceGetAttribute
                    0.00%  23.800us         1  23.800us  23.800us  23.800us  cudaGetDeviceProperties
                    0.00%  11.400us         3  3.8000us     100ns  9.4000us  cuDeviceGetCount
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 20
==31344== NVPROF is profiling process 31344, command: ./Cuda.exe 20
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (3511876), grid(8, 937), block(128, 1)
[host] Arrays do not match!
[host] host 3409268461914 gpu     0 at current 0
==31344== Profiling application: ./Cuda.exe 20
==31344== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==31344== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   48.79%  197.087s    171717  1.1477ms  1.1289ms  15.547ms  [CUDA memcpy HtoD]
                   46.53%  187.935s    171717  1.0944ms  17.504us  12.024ms  [CUDA memcpy DtoH]
                    4.68%  18.9113s    171716  110.13us  74.239us  1.0139ms  addRange(int*, int, int, int, int, int, int, int)
                    0.00%  750.52us         1  750.52us  750.52us  750.52us  sumRange(int*, __int64*, int, int, int, int, int, int)
      API calls:   79.14%  390.329s    343434  1.1365ms  69.500us  99.123ms  cudaMemcpy
                   20.16%  99.4201s    171717  578.98us  68.800us  7.5126ms  cudaDeviceSynchronize
                    0.68%  3.33226s    171717  19.405us  10.400us  4.7635ms  cudaLaunchKernel
                    0.02%  91.014ms         1  91.014ms  91.014ms  91.014ms  cudaSetDevice
                    0.00%  24.265ms         1  24.265ms  24.265ms  24.265ms  cuDevicePrimaryCtxRelease
                    0.00%  400.10us         2  200.05us  116.60us  283.50us  cudaFree
                    0.00%  309.60us         2  154.80us  66.800us  242.80us  cudaMalloc
                    0.00%  60.000us         1  60.000us  60.000us  60.000us  cuLibraryUnload
                    0.00%  32.000us       114     280ns       0ns  13.000us  cuDeviceGetAttribute
                    0.00%  3.6000us         1  3.6000us  3.6000us  3.6000us  cudaGetDeviceProperties
                    0.00%  2.8000us         3     933ns     100ns  2.5000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid
*/
