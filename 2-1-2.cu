#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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
void checkResultInt(int* h_data, int *d_data, const int n) {
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

__global__ void sumArrayElementK(int* g_idata, int* g_odata, unsigned int n, int s, int e, int k) {
    unsigned int tid = threadIdx.x;
    int* idata = g_idata + blockIdx.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    // branch divergence!!!
    if (s <= idx && idx<= e) g_idata[idx] *= k;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] %s starting transpose at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    char in_filename[100];
    sprintf(in_filename, "%s.in", argv[1]);
    freopen(in_filename, "r", stdin);

    int n, s, e, k;
    scanf("%d", &n);
    int nbytes = n * sizeof(int);
    int* h_A = (int*)malloc(nbytes);
    for (int i = 0; i < n; i++) {
        scanf("%d", h_A + i);
    }
    scanf("%d%d%d", &s, &e, &k);

    int h_answer = 0;
    for (int i = 0; i < n; i++) {
        if (s <= i && i <= e) h_answer += h_A[i] * k;
        else h_answer += h_A[i];
    }
    printf("[host] host answer : %d\n", h_answer);

    int* d_idata, * d_odata;
    int blocksize = 512;
    int size = n;
    dim3 block(blocksize, 1);
    dim3 grid((nbytes + blocksize - 1) / blocksize, 1);
    CHECK(cudaMalloc((void**)&d_idata, nbytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));
    cudaMemcpy(d_idata, h_A, nbytes, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), gird(%d), block(%d)\n", size, grid.x, block.x);
    sumArrayElementK << <grid, block >> > (d_idata, d_odata, n, s, e, k);
    int* h_odata = (int*)malloc(grid.x * sizeof(int));
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    int d_answer = 0;
    for (int i = 0; i < grid.x; i++) d_answer += h_odata[i];
    printf("[host] device answer : %d\n", d_answer);
    checkResultInt(&h_answer, &d_answer, 1);

    // memory free
    free(h_A);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==3924== NVPROF is profiling process 3924, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 53697915
[host] datasize (10000), gird(79), block(512)
[host] device answer : 53697915
[host] Arrays match.

==3924== Profiling application: ./Cuda.exe 1
==3924== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==3924== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.56%  10.240us         1  10.240us  10.240us  10.240us  sumArrayElementK(int*, int*, unsigned int, int, int, int)
                   32.81%  6.0480us         1  6.0480us  6.0480us  6.0480us  [CUDA memcpy HtoD]
                   11.63%  2.1440us         1  2.1440us  2.1440us  2.1440us  [CUDA memcpy DtoH]
      API calls:   46.64%  95.833ms         1  95.833ms  95.833ms  95.833ms  cudaLaunchKernel
                   36.44%  74.869ms         1  74.869ms  74.869ms  74.869ms  cudaSetDevice
                   16.37%  33.647ms         1  33.647ms  33.647ms  33.647ms  cuDevicePrimaryCtxRelease
                    0.26%  535.00us         2  267.50us  20.500us  514.50us  cudaFree
                    0.15%  305.60us         2  152.80us  4.6000us  301.00us  cudaMalloc
                    0.09%  177.70us         2  88.850us  64.600us  113.10us  cudaMemcpy
                    0.03%  54.100us       114     474ns       0ns  29.500us  cuDeviceGetAttribute
                    0.02%  43.600us         1  43.600us  43.600us  43.600us  cuLibraryUnload
                    0.00%  5.0000us         1  5.0000us  5.0000us  5.0000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==13332== NVPROF is profiling process 13332, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 203276154
[host] datasize (10000), gird(79), block(512)
[host] device answer : 203276154
[host] Arrays match.

==13332== Profiling application: ./Cuda.exe 2
==13332== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==13332== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.10%  10.368us         1  10.368us  10.368us  10.368us  sumArrayElementK(int*, int*, unsigned int, int, int, int)
                   32.49%  6.1130us         1  6.1130us  6.1130us  6.1130us  [CUDA memcpy HtoD]
                   12.41%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
      API calls:   70.92%  72.250ms         1  72.250ms  72.250ms  72.250ms  cudaSetDevice
                   26.88%  27.382ms         1  27.382ms  27.382ms  27.382ms  cuDevicePrimaryCtxRelease
                    1.12%  1.1428ms         1  1.1428ms  1.1428ms  1.1428ms  cudaLaunchKernel
                    0.39%  394.80us         2  197.40us  40.800us  354.00us  cudaFree
                    0.38%  383.00us         2  191.50us  7.7000us  375.30us  cudaMalloc
                    0.21%  209.10us         2  104.55us  69.100us  140.00us  cudaMemcpy
                    0.04%  44.700us         1  44.700us  44.700us  44.700us  cuLibraryUnload
                    0.04%  40.600us       114     356ns       0ns  23.100us  cuDeviceGetAttribute
                    0.02%  21.900us         2  10.950us     100ns  21.800us  cuDeviceGet
                    0.01%  5.3000us         1  5.3000us  5.3000us  5.3000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetUuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==19584== NVPROF is profiling process 19584, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : 144294235
[host] datasize (10000), gird(79), block(512)
[host] device answer : 144294235
[host] Arrays match.

==19584== Profiling application: ./Cuda.exe 3
==19584== Warning: 36 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==19584== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.03%  10.336us         1  10.336us  10.336us  10.336us  sumArrayElementK(int*, int*, unsigned int, int, int, int)
                   33.56%  6.3040us         1  6.3040us  6.3040us  6.3040us  [CUDA memcpy HtoD]
                   11.41%  2.1440us         1  2.1440us  2.1440us  2.1440us  [CUDA memcpy DtoH]
      API calls:   70.38%  70.508ms         1  70.508ms  70.508ms  70.508ms  cudaSetDevice
                   26.89%  26.937ms         1  26.937ms  26.937ms  26.937ms  cuDevicePrimaryCtxRelease
                    1.21%  1.2144ms         1  1.2144ms  1.2144ms  1.2144ms  cudaLaunchKernel
                    0.84%  846.20us         2  423.10us  26.100us  820.10us  cudaFree
                    0.32%  321.00us         2  160.50us  6.3000us  314.70us  cudaMalloc
                    0.20%  201.00us         2  100.50us  59.200us  141.80us  cudaMemcpy
                    0.09%  86.800us         1  86.800us  86.800us  86.800us  cuLibraryUnload
                    0.05%  54.600us       114     478ns       0ns  36.900us  cuDeviceGetAttribute
                    0.00%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  1.9000us         3     633ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
