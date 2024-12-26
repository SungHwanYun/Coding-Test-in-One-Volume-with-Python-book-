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
==28472== NVPROF is profiling process 28472, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (40000), grid(1, 100), block(128, 1)
[host] Arrays match.

==28472== Profiling application: ./Cuda.exe 1
==28472== Warning: 24 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28472== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.18%  61.822ms     10000  6.1820us  1.1840us  24.704us  [CUDA memcpy DtoH]
                   31.93%  50.386ms      4971  10.136us  8.1600us  11.712us  sumRange(int*, __int64*, int, int, int, int, int, int)
                   20.31%  32.047ms      5030  6.3710us  5.6950us  31.072us  [CUDA memcpy HtoD]
                    8.58%  13.539ms      5029  2.6920us  2.1110us  3.2960us  addRange(int*, int, int, int, int, int, int, int)
      API calls:   40.07%  308.32ms     15030  20.513us  6.5000us  347.60us  cudaMemcpy
                   24.72%  190.19ms     10000  19.018us  2.9000us  79.100us  cudaDeviceSynchronize
                   16.52%  127.10ms         1  127.10ms  127.10ms  127.10ms  cudaSetDevice
                   15.09%  116.09ms     10000  11.608us  6.0000us  3.2183ms  cudaLaunchKernel
                    3.50%  26.927ms         1  26.927ms  26.927ms  26.927ms  cuDevicePrimaryCtxRelease
                    0.05%  416.80us         2  208.40us  6.4000us  410.40us  cudaMalloc
                    0.04%  287.70us         2  143.85us  36.500us  251.20us  cudaFree
                    0.01%  53.700us         1  53.700us  53.700us  53.700us  cuLibraryUnload
                    0.01%  42.100us       114     369ns       0ns  24.400us  cuDeviceGetAttribute
                    0.00%  4.2000us         1  4.2000us  4.2000us  4.2000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==24560== NVPROF is profiling process 24560, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (40000), grid(1, 100), block(128, 1)
[host] Arrays match.

==24560== Profiling application: ./Cuda.exe 2
==24560== Warning: 3 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24560== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.01%  60.914ms     10000  6.0910us  1.1840us  25.696us  [CUDA memcpy DtoH]
                   32.00%  49.965ms      5003  9.9870us  8.5110us  12.192us  sumRange(int*, __int64*, int, int, int, int, int, int)
                   20.45%  31.935ms      4998  6.3890us  5.7270us  25.696us  [CUDA memcpy HtoD]
                    8.54%  13.334ms      4997  2.6680us  2.1440us  5.0240us  addRange(int*, int, int, int, int, int, int, int)
      API calls:   43.74%  311.08ms     14998  20.741us  7.1000us  199.00us  cudaMemcpy
                   25.82%  183.66ms     10000  18.365us  2.4000us  185.60us  cudaDeviceSynchronize
                   16.49%  117.29ms     10000  11.728us  6.1000us  1.0993ms  cudaLaunchKernel
                    9.65%  68.641ms         1  68.641ms  68.641ms  68.641ms  cudaSetDevice
                    4.19%  29.810ms         1  29.810ms  29.810ms  29.810ms  cuDevicePrimaryCtxRelease
                    0.06%  424.80us         2  212.40us  17.600us  407.20us  cudaFree
                    0.03%  240.80us         2  120.40us  5.6000us  235.20us  cudaMalloc
                    0.01%  36.200us         1  36.200us  36.200us  36.200us  cuLibraryUnload
                    0.00%  29.100us       114     255ns       0ns  3.8000us  cuDeviceGetAttribute
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cuDeviceTotalMem
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.2000us         2     600ns     200ns  1.0000us  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==21648== NVPROF is profiling process 21648, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] datasize (40000), grid(1, 100), block(128, 1)
[host] Arrays match.

==21648== Profiling application: ./Cuda.exe 3
==21648== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21648== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.95%  60.375ms     10000  6.0370us  1.1840us  26.176us  [CUDA memcpy DtoH]
                   32.48%  50.345ms      5098  9.8750us  8.1920us  11.968us  sumRange(int*, __int64*, int, int, int, int, int, int)
                   20.24%  31.372ms      4903  6.3980us  5.6960us  21.888us  [CUDA memcpy HtoD]
                    8.32%  12.903ms      4902  2.6320us  2.1120us  4.0640us  addRange(int*, int, int, int, int, int, int, int)
      API calls:   43.41%  304.01ms     14903  20.399us  7.1000us  246.60us  cudaMemcpy
                   26.14%  183.03ms     10000  18.302us  2.6000us  280.70us  cudaDeviceSynchronize
                   16.64%  116.53ms     10000  11.652us  6.1000us  1.2626ms  cudaLaunchKernel
                    9.92%  69.464ms         1  69.464ms  69.464ms  69.464ms  cudaSetDevice
                    3.81%  26.692ms         1  26.692ms  26.692ms  26.692ms  cuDevicePrimaryCtxRelease
                    0.04%  254.10us         2  127.05us  19.700us  234.40us  cudaFree
                    0.03%  208.10us         2  104.05us  6.7000us  201.40us  cudaMalloc
                    0.00%  34.000us         1  34.000us  34.000us  34.000us  cuLibraryUnload
                    0.00%  20.300us       114     178ns       0ns  2.3000us  cuDeviceGetAttribute
                    0.00%  3.3000us         1  3.3000us  3.3000us  3.3000us  cudaGetDeviceProperties
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuModuleGetLoadingMode
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
