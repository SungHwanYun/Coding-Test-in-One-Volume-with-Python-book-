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
    CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    for (auto &b: B) {
        int op = get<0>(b), sy = get<1>(b), sx = get<2>(b), ey = get<3>(b), ex = get<4>(b), k = get<5>(b);
        if (op == 1) {
            //CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
            addRange << <grid, block >> > (d_A, nx, ny, sy, sx, ey, ex, k);
            cudaDeviceSynchronize();
            //CHECK(cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost));
        }
        else {
            //CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
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
==23556== NVPROF is profiling process 23556, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (32), grid(1, 2), block(128, 1)
[host] Arrays match.

==23556== Profiling application: ./Cuda.exe 1
==23556== Warning: 11 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==23556== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.62%  12.544us         4  3.1360us  3.0080us  3.5200us  addRange(__int64*, int, int, int, int, int, int, int)
                   31.81%  7.5840us         1  7.5840us  7.5840us  7.5840us  sumRange(__int64*, __int64*, int, int, int, int, int, int)
                    9.80%  2.3370us         1  2.3370us  2.3370us  2.3370us  [CUDA memcpy DtoH]
                    5.77%  1.3760us         2     688ns     352ns  1.0240us  [CUDA memcpy HtoD]
      API calls:   76.22%  85.166ms         1  85.166ms  85.166ms  85.166ms  cudaSetDevice
                   21.91%  24.483ms         1  24.483ms  24.483ms  24.483ms  cuDevicePrimaryCtxRelease
                    1.00%  1.1227ms         5  224.54us  7.1000us  1.0735ms  cudaLaunchKernel
                    0.30%  333.60us         2  166.80us  16.300us  317.30us  cudaFree
                    0.18%  206.10us         3  68.700us  41.500us  112.40us  cudaMemcpy
                    0.13%  146.70us         2  73.350us  17.700us  129.00us  cudaMalloc
                    0.11%  126.20us         1  126.20us  126.20us  126.20us  cuLibraryUnload
                    0.07%  74.200us       114     650ns       0ns  23.400us  cuDeviceGetAttribute
                    0.05%  58.600us         5  11.720us  8.5000us  15.100us  cudaDeviceSynchronize
                    0.00%  3.4000us         1  3.4000us  3.4000us  3.4000us  cudaGetDeviceProperties
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuModuleGetLoadingMode
                    0.00%  2.3000us         3     766ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  1.5000us         2     750ns     100ns  1.4000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 12
==11664== NVPROF is profiling process 11664, command: ./Cuda.exe 12
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (631688), grid(3, 281), block(128, 1)
[host] Arrays match.

==11664== Profiling application: ./Cuda.exe 12
==11664== Warning: 26 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==11664== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.96%  1.28516s    119970  10.712us  9.5680us  25.632us  addRange(__int64*, int, int, int, int, int, int, int)
                    0.04%  457.98us         2  228.99us  220.58us  237.41us  [CUDA memcpy HtoD]
                    0.00%  56.000us         1  56.000us  56.000us  56.000us  sumRange(__int64*, __int64*, int, int, int, int, int, int)
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  [CUDA memcpy DtoH]
      API calls:   64.52%  2.10336s    119971  17.532us  2.4000us  303.00us  cudaDeviceSynchronize
                   32.77%  1.06839s    119971  8.9050us  5.2000us  1.3004ms  cudaLaunchKernel
                    1.97%  64.180ms         1  64.180ms  64.180ms  64.180ms  cudaSetDevice
                    0.71%  23.077ms         1  23.077ms  23.077ms  23.077ms  cuDevicePrimaryCtxRelease
                    0.01%  363.80us         3  121.27us  79.300us  148.40us  cudaMemcpy
                    0.01%  290.40us         2  145.20us  18.800us  271.60us  cudaFree
                    0.01%  231.40us         2  115.70us  6.1000us  225.30us  cudaMalloc
                    0.00%  56.100us       114     492ns       0ns  35.900us  cuDeviceGetAttribute
                    0.00%  38.100us         1  38.100us  38.100us  38.100us  cuLibraryUnload
                    0.00%  4.3000us         1  4.3000us  4.3000us  4.3000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     700ns         2     350ns     100ns     600ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 20
==25028== NVPROF is profiling process 25028, command: ./Cuda.exe 20
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (7023752), grid(8, 937), block(128, 1)
[host] Arrays match.

==25028== Profiling application: ./Cuda.exe 20
==25028== Warning: 33 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==25028== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.88%  14.7695s    171716  86.011us  73.312us  2.6525ms  addRange(__int64*, int, int, int, int, int, int, int)
                    0.12%  17.792ms         2  8.8960ms  8.8913ms  8.9007ms  [CUDA memcpy HtoD]
                    0.00%  496.38us         1  496.38us  496.38us  496.38us  sumRange(__int64*, __int64*, int, int, int, int, int, int)
                    0.00%  17.952us         1  17.952us  17.952us  17.952us  [CUDA memcpy DtoH]
      API calls:   90.22%  16.3204s    171717  95.042us  3.6000us  2.6638ms  cudaDeviceSynchronize
                    8.28%  1.49803s    171717  8.7230us  5.3000us  2.9953ms  cudaLaunchKernel
                    0.70%  126.79ms         1  126.79ms  126.79ms  126.79ms  cudaSetDevice
                    0.63%  114.63ms         3  38.211ms  96.900us  105.80ms  cudaMemcpy
                    0.16%  29.040ms         1  29.040ms  29.040ms  29.040ms  cuDevicePrimaryCtxRelease
                    0.00%  505.90us         2  252.95us  187.00us  318.90us  cudaFree
                    0.00%  330.80us         2  165.40us  73.700us  257.10us  cudaMalloc
                    0.00%  39.000us         1  39.000us  39.000us  39.000us  cuLibraryUnload
                    0.00%  19.100us       114     167ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
