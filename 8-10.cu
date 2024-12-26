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

#define DIM 128
__global__ void addRange(long long* d_A, const int nx, int ny, int sy, int sx, int ey, int ex, int k) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    // boundary check
    if (sx <= ix && ix <= ex && sy <= iy && iy <= ey) {
        d_A[idx] += k;
    }
}

__global__ void sumRange(long long* d_A, long long* d_B, const int nx, int ny, int sy, int sx, int ey, int ex) {
    __shared__ long long smem[DIM];
    unsigned int tid = threadIdx.x;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix >= nx || iy >= ny) return;
    if (sx <= ix && ix <= ex && sy <= iy && iy <= ey) smem[tid] = d_A[idx];
    else smem[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && ix + stride < nx) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_B[blockIdx.y * gridDim.x + blockIdx.x] = smem[0];
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

    int n, m; cin >> n >> m;
    vector<vector<long long>>A(n, vector<long long>(n));
    long long* h_A = (long long*)malloc(n * n * sizeof(long long));
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
    int nbytes = n * n * sizeof(long long);
    int kbytes = k * sizeof(long long);

    /* device code */
    long long* d_A; long long* d_B;
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
            CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
            sumRange << <grid, block >> > (d_A, d_B, nx, ny, sy, sx, ey, ex);
            cudaDeviceSynchronize();
            CHECK(cudaMemcpy(tmp, d_B, grid.x * grid.y * sizeof(long long), cudaMemcpyDeviceToHost));
            long long x = 0;
            for (int i = 0; i < grid.x; i++) for (int j = 0; j < grid.y; j++) {
                x += tmp[j * grid.x + i];
            }
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
==7040== NVPROF is profiling process 7040, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (32), grid(1, 2), block(128, 1)
[host] Arrays match.

==7040== Profiling application: ./Cuda.exe 1
==7040== Warning: 20 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==7040== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.55%  12.768us         4  3.1920us  2.9760us  3.7760us  addRange(__int64*, int, int, int, int, int, int, int)
                   26.83%  8.4470us         5  1.6890us  1.4710us  2.1440us  [CUDA memcpy DtoH]
                   23.98%  7.5520us         1  7.5520us  7.5520us  7.5520us  sumRange(__int64*, __int64*, int, int, int, int, int, int)
                    8.64%  2.7200us         6     453ns     320ns     991ns  [CUDA memcpy HtoD]
      API calls:   74.17%  81.976ms         1  81.976ms  81.976ms  81.976ms  cudaSetDevice
                   24.20%  26.750ms         1  26.750ms  26.750ms  26.750ms  cuDevicePrimaryCtxRelease
                    0.85%  942.70us         5  188.54us  8.0000us  904.90us  cudaLaunchKernel
                    0.28%  307.10us        11  27.918us  4.9000us  104.10us  cudaMemcpy
                    0.21%  229.30us         2  114.65us  8.4000us  220.90us  cudaFree
                    0.11%  123.40us         2  61.700us  4.6000us  118.80us  cudaMalloc
                    0.09%  98.100us         1  98.100us  98.100us  98.100us  cuLibraryUnload
                    0.04%  47.900us         5  9.5800us  8.2000us  12.700us  cudaDeviceSynchronize
                    0.03%  33.100us       114     290ns       0ns  13.500us  cuDeviceGetAttribute
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  1.9000us         3     633ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 12
==25864== NVPROF is profiling process 25864, command: ./Cuda.exe 12
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (631688), grid(3, 281), block(128, 1)
[host] Arrays match.

==25864== Profiling application: ./Cuda.exe 12
==25864== Warning: 1 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==25864== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.24%  24.5489s    119972  204.62us  202.65us  512.22us  [CUDA memcpy HtoD]
                   47.04%  22.9867s    119971  191.60us  1.6960us  250.46us  [CUDA memcpy DtoH]
                    2.71%  1.32627s    119970  11.055us  9.9830us  27.936us  addRange(__int64*, int, int, int, int, int, int, int)
                    0.00%  56.575us         1  56.575us  56.575us  56.575us  sumRange(__int64*, __int64*, int, int, int, int, int, int)
      API calls:   55.67%  41.7661s    239943  174.07us  18.500us  4.3265ms  cudaMemcpy
                   42.08%  31.5692s    119971  263.14us  8.6000us  11.465ms  cudaDeviceSynchronize
                    2.08%  1.56387s    119971  13.035us  7.4000us  1.2777ms  cudaLaunchKernel
                    0.13%  94.425ms         1  94.425ms  94.425ms  94.425ms  cudaSetDevice
                    0.04%  33.442ms         1  33.442ms  33.442ms  33.442ms  cuDevicePrimaryCtxRelease
                    0.00%  310.30us         2  155.15us  17.700us  292.60us  cudaFree
                    0.00%  252.40us         2  126.20us  7.3000us  245.10us  cudaMalloc
                    0.00%  41.200us         1  41.200us  41.200us  41.200us  cuLibraryUnload
                    0.00%  32.300us       114     283ns     100ns  5.8000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cuModuleGetLoadingMode
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.3000us         3     766ns       0ns  1.8000us  cuDeviceGetCount
                    0.00%  1.3000us         2     650ns     200ns  1.1000us  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 20
==14280== NVPROF is profiling process 14280, command: ./Cuda.exe 20
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (7023752), grid(8, 937), block(128, 1)
[host] Arrays match.

==14280== Profiling application: ./Cuda.exe 20
==14280== Warning: 35 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==14280== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.34%  394.983s    171718  2.3002ms  2.2542ms  14.653ms  [CUDA memcpy HtoD]
                   48.20%  385.830s    171717  2.2469ms  17.312us  12.331ms  [CUDA memcpy DtoH]
                    2.47%  19.7386s    171716  114.95us  74.240us  6.4464ms  addRange(__int64*, int, int, int, int, int, int, int)
                    0.00%  683.35us         1  683.35us  683.35us  683.35us  sumRange(__int64*, __int64*, int, int, int, int, int, int)
      API calls:   86.72%  795.974s    343435  2.3177ms  191.40us  104.19ms  cudaMemcpy
                   12.68%  116.392s    171717  677.81us  89.500us  7.4705ms  cudaDeviceSynchronize
                    0.58%  5.36208s    171717  31.226us  11.100us  2.9365ms  cudaLaunchKernel
                    0.01%  79.122ms         1  79.122ms  79.122ms  79.122ms  cudaSetDevice
                    0.00%  29.167ms         1  29.167ms  29.167ms  29.167ms  cuDevicePrimaryCtxRelease
                    0.00%  681.20us         2  340.60us  313.20us  368.00us  cudaMalloc
                    0.00%  606.50us         2  303.25us  140.80us  465.70us  cudaFree
                    0.00%  60.900us         1  60.900us  60.900us  60.900us  cuLibraryUnload
                    0.00%  21.200us       114     185ns       0ns  3.2000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.6000us         3     866ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
