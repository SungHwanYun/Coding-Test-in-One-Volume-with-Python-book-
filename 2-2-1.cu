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

__global__ void getStringEvenPosition(char* g_idata, char* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n) return;

    // branch divergence!!!
    if (idx % 2 == 1) {
        g_odata[idx / 2] = g_idata[idx];
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

    string A; cin >> A;
    int n = (int)A.length();
    string B;
    for (int i = 1; i < A.size(); i += 2) {
        B = B + A[i];
    }
    int m = (int)B.length();
    char* h_answer = (char *)malloc((m + 1) * sizeof(char));
    memcpy(h_answer, B.c_str(), m); h_answer[m] = 0;
    printf("[host] host answer : %s\n", h_answer);

    /* host - device code */
    int nbytes = n * sizeof(char);
    char* h_A = (char*)malloc(nbytes + 1);
    memcpy(h_A, A.c_str(), n);

    /* device code */
    char* d_A, *d_odata;
    dim3 block(32, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    CHECK(cudaMalloc((void**)&d_A, nbytes + 1));
    CHECK(cudaMalloc((void**)&d_odata, nbytes + 1));
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", nbytes, grid.x, grid.y, block.x, block.y);
    getStringEvenPosition << <grid, block >> > (d_A, d_odata, n);
    char* d_answer = (char*)malloc(m + 1);
    cudaMemcpy(d_answer, d_odata, m, cudaMemcpyDeviceToHost);
    d_answer[m] = 0;
    printf("[host] device answer : %s\n", d_answer);
    checkResult<char>(h_answer, d_answer, m);

    // memory free
    free(h_A); free(h_answer); free(d_answer);
    cudaFree(d_A); cudaFree(d_odata);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==8948== NVPROF is profiling process 8948, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : iaBhGbcbPomlFkHSuTHOjWeCnWojkGJDrEXDUHTUoEoKdYQPmARpYmOYyRSpORIGOBqYgDkLmPHGOszsTVVgnNHiLYuXaAHZmxiQVxmXhtEhcuoMytxuavhbQqAgLmQvncrIXzRRBuOHoEhglIcTzthvLqCwJDCDNPePSseXbzppNRUKymDLJmWuwbaLEoiYxlKdjtnMKKZqbxksoMEbIidJRqfOJkSndenOBUaoyCHimRRxYFSGNdgjgVxnzXYejPeHwLOkDkBdXPIZMNNELYJMkLwYelLgLXePJWsSyZicoVDlWstWaNjwVEtFaHEZreDeGfSwSyPGPiggGrfSCqqgAgPcAUFtJYFYlQmGbGdXIPmWeRCesLLuHsabnUoNERASSsHKBihotXOXMWKKPOFxEbDpbjQYQGtqXniWJcGToazgDOTQLxEabZMCQuCGeKrPCROVqKRaEYdvPqWuATAfijfLvAQfZRGaWPuLELsYVdwuMJGv
[host] datasize (1000), grid(32, 1), block(32, 1)
[host] device answer : iaBhGbcbPomlFkHSuTHOjWeCnWojkGJDrEXDUHTUoEoKdYQPmARpYmOYyRSpORIGOBqYgDkLmPHGOszsTVVgnNHiLYuXaAHZmxiQVxmXhtEhcuoMytxuavhbQqAgLmQvncrIXzRRBuOHoEhglIcTzthvLqCwJDCDNPePSseXbzppNRUKymDLJmWuwbaLEoiYxlKdjtnMKKZqbxksoMEbIidJRqfOJkSndenOBUaoyCHimRRxYFSGNdgjgVxnzXYejPeHwLOkDkBdXPIZMNNELYJMkLwYelLgLXePJWsSyZicoVDlWstWaNjwVEtFaHEZreDeGfSwSyPGPiggGrfSCqqgAgPcAUFtJYFYlQmGbGdXIPmWeRCesLLuHsabnUoNERASSsHKBihotXOXMWKKPOFxEbDpbjQYQGtqXniWJcGToazgDOTQLxEabZMCQuCGeKrPCROVqKRaEYdvPqWuATAfijfLvAQfZRGaWPuLELsYVdwuMJGv
[host] Arrays match.

==8948== Profiling application: ./Cuda.exe 1
==8948== Warning: 16 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==8948== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.48%  3.9040us         1  3.9040us  3.9040us  3.9040us  getStringEvenPosition(char*, char*, unsigned int)
                   28.41%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
                   24.12%  1.9830us         1  1.9830us  1.9830us  1.9830us  [CUDA memcpy HtoD]
      API calls:   78.80%  102.89ms         1  102.89ms  102.89ms  102.89ms  cudaSetDevice
                   19.89%  25.971ms         1  25.971ms  25.971ms  25.971ms  cuDevicePrimaryCtxRelease
                    0.86%  1.1253ms         1  1.1253ms  1.1253ms  1.1253ms  cudaLaunchKernel
                    0.13%  167.40us         2  83.700us  11.500us  155.90us  cudaFree
                    0.12%  156.20us         2  78.100us  6.3000us  149.90us  cudaMalloc
                    0.11%  139.40us         2  69.700us  44.400us  95.000us  cudaMemcpy
                    0.06%  84.800us       114     743ns       0ns  45.700us  cuDeviceGetAttribute
                    0.02%  21.000us         1  21.000us  21.000us  21.000us  cuLibraryUnload
                    0.00%  6.5000us         1  6.5000us  6.5000us  6.5000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  2.2000us         3     733ns       0ns  1.6000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     700ns         2     350ns     100ns     600ns  cuDeviceGet
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==24956== NVPROF is profiling process 24956, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : SNcSBTmphBXgocAjRuMlIpdKhezxBfkPJafnenBUvqymmmZAvTUgjZpHXDcVCtbjjaIEcQSboagEyddersKDhuvgfAvRJQHvisHZlAJqrQUWgPOxCdzmhTgZNVAmjDCPoxlKMZfZYVeoZarDzTONLUqrRnPOiSnAsCZERevRAzDvMpxamITZrtDgsslHsJvhSwPaBLZtFMWqxtpNfqqlKeWrlKVkVZtuOiUgSFZNFLTNIAYcrUwqDYYVopnejUMcNtWSaHPSkVaNywxAdfMKqyDXNvgrAAXNyDTtVBEWAkcdJEJUthunvQxqBeSxfRLzJWCURbLvnOvBXRWILUWoTHtlrVzXENyNXhcvMisdMihVoqSkkgJFQRkBrwmabOPAiAcjVBcmHdCtCSUBeXhLpUIoFwoYVzemUEGiggOEhgoCrYIZAjQxvbzsSLHrFsfVcAGGJDWqUgpDShzOLXOnkTEQpHBynifljgQLJTQcuAgUHbooYTNJ
[host] datasize (1000), grid(32, 1), block(32, 1)
[host] device answer : SNcSBTmphBXgocAjRuMlIpdKhezxBfkPJafnenBUvqymmmZAvTUgjZpHXDcVCtbjjaIEcQSboagEyddersKDhuvgfAvRJQHvisHZlAJqrQUWgPOxCdzmhTgZNVAmjDCPoxlKMZfZYVeoZarDzTONLUqrRnPOiSnAsCZERevRAzDvMpxamITZrtDgsslHsJvhSwPaBLZtFMWqxtpNfqqlKeWrlKVkVZtuOiUgSFZNFLTNIAYcrUwqDYYVopnejUMcNtWSaHPSkVaNywxAdfMKqyDXNvgrAAXNyDTtVBEWAkcdJEJUthunvQxqBeSxfRLzJWCURbLvnOvBXRWILUWoTHtlrVzXENyNXhcvMisdMihVoqSkkgJFQRkBrwmabOPAiAcjVBcmHdCtCSUBeXhLpUIoFwoYVzemUEGiggOEhgoCrYIZAjQxvbzsSLHrFsfVcAGGJDWqUgpDShzOLXOnkTEQpHBynifljgQLJTQcuAgUHbooYTNJ
[host] Arrays match.

==24956== Profiling application: ./Cuda.exe 2
==24956== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24956== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   45.14%  3.7120us         1  3.7120us  3.7120us  3.7120us  getStringEvenPosition(char*, char*, unsigned int)
                   29.57%  2.4320us         1  2.4320us  2.4320us  2.4320us  [CUDA memcpy DtoH]
                   25.29%  2.0800us         1  2.0800us  2.0800us  2.0800us  [CUDA memcpy HtoD]
      API calls:   76.58%  73.901ms         1  73.901ms  73.901ms  73.901ms  cudaSetDevice
                   21.11%  20.367ms         1  20.367ms  20.367ms  20.367ms  cuDevicePrimaryCtxRelease
                    1.52%  1.4648ms         1  1.4648ms  1.4648ms  1.4648ms  cudaLaunchKernel
                    0.29%  276.80us         2  138.40us  90.400us  186.40us  cudaMemcpy
                    0.23%  219.10us         2  109.55us  34.600us  184.50us  cudaFree
                    0.22%  212.10us         2  106.05us  6.5000us  205.60us  cudaMalloc
                    0.02%  22.900us         1  22.900us  22.900us  22.900us  cuLibraryUnload
                    0.02%  21.300us       114     186ns       0ns  3.9000us  cuDeviceGetAttribute
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cudaGetDeviceProperties
                    0.00%  1.8000us         3     600ns       0ns  1.6000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==21876== NVPROF is profiling process 21876, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host answer : fYdndDzZHilVyEuYLyOvNTVSduGZlnumdKuyFzIcBVLhweqOWFSENstfEKvwfiMwQakONpiNAyeriHrDguhodHYkSEgqNoHYbZWpJGEcDRtAEGQCTOBWuVVeNpBxPAaNqeYPqRyxCXMDDbtzaoeSaVJisnijyDKCtlyiXJVNFGnQRvDGeAGBHPhFsUIjVrNIQChcsFvHxQfrpUJDdrXOUHKyPHKeAutRAqgIJaGLktrgIUuDdkUJiGPuLVMCEPpXcIPqjyJawrfoHDdtBPtejqsfcYydreBlgQyuAEdWFwIWrfjtuUnGtmSnwuJincqAtPSlmUwtdLYivauUWUwfzfANhyTyLIrKyHWqVSMeqchhArlxvJWHOeyVSECHMuYRjKmJbbIyTjrNMCYIQZlOYgYQPtCrJuNQkjmSqaREcpMkNUkNCgDKPsrdIUpVihXIhfbqhjlYuZuCmzrtDcGRuDNwuVBifyJvLIsgkjyCcvsdoNfapntu
[host] datasize (1000), grid(32, 1), block(32, 1)
[host] device answer : fYdndDzZHilVyEuYLyOvNTVSduGZlnumdKuyFzIcBVLhweqOWFSENstfEKvwfiMwQakONpiNAyeriHrDguhodHYkSEgqNoHYbZWpJGEcDRtAEGQCTOBWuVVeNpBxPAaNqeYPqRyxCXMDDbtzaoeSaVJisnijyDKCtlyiXJVNFGnQRvDGeAGBHPhFsUIjVrNIQChcsFvHxQfrpUJDdrXOUHKyPHKeAutRAqgIJaGLktrgIUuDdkUJiGPuLVMCEPpXcIPqjyJawrfoHDdtBPtejqsfcYydreBlgQyuAEdWFwIWrfjtuUnGtmSnwuJincqAtPSlmUwtdLYivauUWUwfzfANhyTyLIrKyHWqVSMeqchhArlxvJWHOeyVSECHMuYRjKmJbbIyTjrNMCYIQZlOYgYQPtCrJuNQkjmSqaREcpMkNUkNCgDKPsrdIUpVihXIhfbqhjlYuZuCmzrtDcGRuDNwuVBifyJvLIsgkjyCcvsdoNfapntu
[host] Arrays match.

==21876== Profiling application: ./Cuda.exe 3
==21876== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21876== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   45.38%  3.6160us         1  3.6160us  3.6160us  3.6160us  getStringEvenPosition(char*, char*, unsigned int)
                   29.32%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
                   25.30%  2.0160us         1  2.0160us  2.0160us  2.0160us  [CUDA memcpy HtoD]
      API calls:   68.75%  68.757ms         1  68.757ms  68.757ms  68.757ms  cudaSetDevice
                   29.13%  29.134ms         1  29.134ms  29.134ms  29.134ms  cuDevicePrimaryCtxRelease
                    1.21%  1.2087ms         1  1.2087ms  1.2087ms  1.2087ms  cudaLaunchKernel
                    0.40%  396.80us         2  198.40us  10.500us  386.30us  cudaFree
                    0.26%  262.40us         2  131.20us  5.6000us  256.80us  cudaMalloc
                    0.15%  146.90us         2  73.450us  59.700us  87.200us  cudaMemcpy
                    0.06%  56.100us       114     492ns       0ns  37.400us  cuDeviceGetAttribute
                    0.02%  24.400us         1  24.400us  24.400us  24.400us  cuLibraryUnload
                    0.01%  9.5000us         1  9.5000us  9.5000us  9.5000us  cuDeviceGetLuid
                    0.00%  3.0000us         1  3.0000us  3.0000us  3.0000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.5000us         1  1.5000us  1.5000us  1.5000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
