#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<bits/stdc++.h>
#include<time.h>
using namespace std;
typedef pair<int, int> pii;

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
            printf("[host] host %5d gpu %5d at current %d\n", h_data[i], d_data[i], i);
            break;
        }
    }
    if (match) printf("[host] Arrays match.\n\n");
}

struct CSRGraph {
    // vertex_number: number of vertex, edge_number: number of edge
    int vertex_number, edge_number;
    int* srcPtrs, * dst;
    void buildData(vector<vector<int>>& edge) {
        int dst_index = 0;
        for (int u = 0; u < vertex_number; u++) {
            srcPtrs[u] = dst_index;
            for (int i = 0; i < edge[u].size(); i++) {
                dst[dst_index++] = edge[u][i];
            }
        }
        if (edge_number != dst_index) exit(-1);
        srcPtrs[vertex_number] = edge_number;
    }
    void printData() {
        printf("vertext_number = %d\n", vertex_number);
        printf("edge_number = %d\n", edge_number);
        for (int u = 0; u <= vertex_number; u++) {
            printf("%d ", srcPtrs[u]);
        }
        printf("\n");
        for (int e = 0; e < edge_number; e++) {
            printf("%d ", dst[e]);
        }
        printf("\n");
    }
};

__device__ CSRGraph csrGraph;
__global__ void bfs_kernel(int* level, int* newVertextVisited, int curLevel) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < csrGraph.vertex_number) {
        if (level[vertex] == curLevel - 1) {
            for (int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                int neighbor = csrGraph.dst[edge];
                if (level[neighbor] == INT_MAX) {
                    level[neighbor] = curLevel;
                    *newVertextVisited = 1;
                }
            }
        }
    }
}

int dr[4] = { -1, 1, 0, 0 }, dc[4] = { 0, 0, -1, 1 };
int in_range(int r, int c) {
    return 0 <= r && r <= 4 && 0 <= c && c <= 4;
}
int get_vertex_number(int r, int c, int n, int m) {
    return r * m + c;
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
    int* h_answer = (int*)malloc(sizeof(int));
    *h_answer = -1;
    int A[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cin >> A[i][j];
        }
    }
    int sr, sc; cin >> sr >> sc;
    int tr, tc;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (A[i][j] == 1) {
                tr = i; tc = j;
            }
        }
    }
    int visited[5][5] = { 0 }, dist[5][5] = { 0 };
    deque<pii> Q;
    Q.push_back({ sr, sc });
    visited[sr][sc] = 1;
    dist[sr][sc] = 0;
    while (!Q.empty()) {
        pii now = Q.front(); Q.pop_front();
        int r = now.first, c = now.second;
        if (r == tr && c == tc) {
            *h_answer = dist[r][c];
            break;
        }
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            if (in_range(nr, nc) && visited[nr][nc] == 0 && A[nr][nc] != -1) {
                Q.push_back({ nr,nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }
    }
    int times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] host time : %d ms\n", times);
    //printf("[host] host answer : %d\n", *h_answer);

    /* host - device code */
    int root = sr * 5 + sc;
    int dst_vertex_number = tr * 5 + tc;
    int edge_number = 0;
    vector<vector<int>> edge(5 * 5, vector<int>());
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (A[i][j] == -1) continue;
            int u = get_vertex_number(i, j, 5, 5);
            for (int k = 0; k < 4; k++) {
                int nr = i + dr[k], nc = j + dc[k];
                if (in_range(nr, nc) && A[nr][nc] != -1) {
                    int v = get_vertex_number(nr, nc, 5, 5);
                    edge[u].push_back(v);
                    edge_number++;
                }
            }
        }
    }

    // structure deep-copy(shallow copy는 에러 발생)
    // https://forums.developer.nvidia.com/t/clean-way-of-copying-a-struct-with-pointers-to-the-gpu/225833/2
    CSRGraph G;
    G.edge_number = edge_number;
    G.vertex_number = 25;
    //CSRGraph G(25, edge_number); // dynamic allocation error
    G.srcPtrs = (int*)malloc((G.vertex_number + 1) * sizeof(int));
    G.dst = (int*)malloc(G.edge_number * sizeof(int));
    G.buildData(edge);

    CSRGraph GDeep;
    GDeep.vertex_number = G.vertex_number;
    GDeep.edge_number = G.edge_number;
    CHECK(cudaMalloc(&(GDeep.srcPtrs), (G.vertex_number + 1) * sizeof(G.srcPtrs[0])));
    CHECK(cudaMalloc(&(GDeep.dst), G.edge_number * sizeof(G.dst[0])));
    CHECK(cudaMemcpy(GDeep.srcPtrs, G.srcPtrs, 
        (G.vertex_number + 1)*sizeof(G.srcPtrs[0]), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(GDeep.dst, G.dst,
        G.edge_number * sizeof(G.dst[0]), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(csrGraph, &GDeep, sizeof(CSRGraph)));

    /* device code */

    int n = 25;
    int* h_level = (int *)malloc(n * sizeof(int));
    int *h_newVertexVisited = (int *)malloc(sizeof(int));
    int* d_level, * d_newVertexVisited;
    CHECK(cudaMalloc((void**)&d_level, n * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_newVertexVisited, sizeof(int)));
    fill(h_level, h_level + n, INT_MAX);
    h_level[root] = 0;
    CHECK(cudaMemcpy(d_level, h_level, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_newVertexVisited, 0, sizeof(int)));

    int* d_answer = (int*)malloc(sizeof(int));
    *d_answer = -1;
    dim3 block(512, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", n, grid.x, grid.y, block.x, block.y);

    start = clock();
    for (int l = 1; ; l++) {
        CHECK(cudaMemset(d_newVertexVisited, 0, sizeof(int)));
        bfs_kernel << <grid, block >> > (d_level, d_newVertexVisited, l);
        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(h_newVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost));
        if (*h_newVertexVisited == 0) break;
        CHECK(cudaMemcpy(h_level, d_level, n * sizeof(int), cudaMemcpyDeviceToHost));
        if (h_level[dst_vertex_number] != INT_MAX) {
            *d_answer = h_level[dst_vertex_number];
            break;
        }
    }
    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] device time : %d ms\n", times);
    
    //printf("[host] device answer : %d\n", *d_answer);
    checkResult<int>(h_answer, d_answer, 1);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==11032== NVPROF is profiling process 11032, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 0 ms
[host] Arrays match.

==11032== Profiling application: ./Cuda.exe 1
==11032== Warning: 35 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==11032== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.57%  18.432us         2  9.2160us  8.8960us  9.5360us  bfs_kernel(int*, int*, int)
                   24.77%  7.0720us         4  1.7680us  1.3760us  2.8480us  [CUDA memcpy DtoH]
                    7.06%  2.0160us         4     504ns     320ns     992ns  [CUDA memcpy HtoD]
                    3.59%  1.0250us         3     341ns     320ns     353ns  [CUDA memset]
      API calls:   71.07%  58.817ms         1  58.817ms  58.817ms  58.817ms  cudaSetDevice
                   26.34%  21.802ms         1  21.802ms  21.802ms  21.802ms  cuDevicePrimaryCtxRelease
                    1.64%  1.3596ms         1  1.3596ms  1.3596ms  1.3596ms  cudaMemcpyToSymbol
                    0.41%  342.40us         7  48.914us  7.9000us  211.30us  cudaMemcpy
                    0.18%  150.50us         1  150.50us  150.50us  150.50us  cuLibraryUnload
                    0.14%  114.10us         4  28.525us  2.5000us  100.90us  cudaMalloc
                    0.11%  87.500us         2  43.750us  21.900us  65.600us  cudaLaunchKernel
                    0.04%  31.500us         2  15.750us  14.700us  16.800us  cudaDeviceSynchronize
                    0.03%  25.500us         3  8.5000us  5.4000us  12.300us  cudaMemset
                    0.02%  19.200us       114     168ns       0ns  3.1000us  cuDeviceGetAttribute
                    0.00%  3.9000us         1  3.9000us  3.9000us  3.9000us  cudaGetDeviceProperties
                    0.00%  2.6000us         3     866ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.3000us         2     650ns       0ns  1.3000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 9
==42316== NVPROF is profiling process 42316, command: ./Cuda.exe 9
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 1 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 0 ms
[host] Arrays match.

==42316== Profiling application: ./Cuda.exe 9
==42316== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==42316== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.52%  11.872us         2  5.9360us  4.5760us  7.2960us  bfs_kernel(int*, int*, int)
                   28.06%  5.7920us         3  1.9300us  1.3440us  2.9440us  [CUDA memcpy DtoH]
                    9.61%  1.9840us         4     496ns     320ns     992ns  [CUDA memcpy HtoD]
                    4.81%     992ns         3     330ns     320ns     352ns  [CUDA memset]
      API calls:   69.81%  59.652ms         1  59.652ms  59.652ms  59.652ms  cudaSetDevice
                   27.20%  23.246ms         1  23.246ms  23.246ms  23.246ms  cuDevicePrimaryCtxRelease
                    1.89%  1.6192ms         1  1.6192ms  1.6192ms  1.6192ms  cudaMemcpyToSymbol
                    0.36%  310.70us         6  51.783us  11.500us  143.70us  cudaMemcpy
                    0.23%  199.00us         4  49.750us  4.0000us  166.00us  cudaMalloc
                    0.22%  186.30us         1  186.30us  186.30us  186.30us  cuLibraryUnload
                    0.12%  105.50us         2  52.750us  17.600us  87.900us  cudaLaunchKernel
                    0.08%  64.600us         3  21.533us  12.400us  31.200us  cudaMemset
                    0.04%  33.300us       114     292ns       0ns  15.600us  cuDeviceGetAttribute
                    0.03%  26.000us         2  13.000us  11.400us  14.600us  cudaDeviceSynchronize
                    0.00%  3.4000us         1  3.4000us  3.4000us  3.4000us  cudaGetDeviceProperties
                    0.00%  2.5000us         3     833ns     100ns  2.1000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 10
==31308== NVPROF is profiling process 31308, command: ./Cuda.exe 10
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 1 ms
[host] Arrays match.

==31308== Profiling application: ./Cuda.exe 10
==31308== Warning: 35 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==31308== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.07%  4.3190us         1  4.3190us  4.3190us  4.3190us  bfs_kernel(int*, int*, int)
                   24.92%  2.3360us         1  2.3360us  2.3360us  2.3360us  [CUDA memcpy DtoH]
                   21.49%  2.0150us         4     503ns     320ns     992ns  [CUDA memcpy HtoD]
                    7.52%     705ns         2     352ns     352ns     353ns  [CUDA memset]
      API calls:   72.09%  68.346ms         1  68.346ms  68.346ms  68.346ms  cudaSetDevice
                   25.43%  24.107ms         1  24.107ms  24.107ms  24.107ms  cuDevicePrimaryCtxRelease
                    1.65%  1.5667ms         1  1.5667ms  1.5667ms  1.5667ms  cudaMemcpyToSymbol
                    0.25%  238.90us         4  59.725us  10.000us  87.600us  cudaMemcpy
                    0.18%  174.20us         4  43.550us  2.8000us  138.10us  cudaMalloc
                    0.18%  168.40us         1  168.40us  168.40us  168.40us  cuLibraryUnload
                    0.13%  119.00us         1  119.00us  119.00us  119.00us  cudaLaunchKernel
                    0.05%  44.700us         2  22.350us  20.000us  24.700us  cudaMemset
                    0.02%  19.600us       114     171ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.01%  11.600us         1  11.600us  11.600us  11.600us  cudaDeviceSynchronize
                    0.00%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 11
==17504== NVPROF is profiling process 17504, command: ./Cuda.exe 11
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 1 ms
[host] Arrays match.

==17504== Profiling application: ./Cuda.exe 11
==17504== Warning: 27 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==17504== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.79%  16.320us         3  5.4400us  4.8640us  5.8240us  bfs_kernel(int*, int*, int)
                   27.95%  8.0320us         5  1.6060us  1.3760us  2.3360us  [CUDA memcpy DtoH]
                   10.69%  3.0720us         4     768ns     352ns  1.3750us  [CUDA memcpy HtoD]
                    4.57%  1.3120us         4     328ns     320ns     352ns  [CUDA memset]
      API calls:   70.37%  57.887ms         1  57.887ms  57.887ms  57.887ms  cudaSetDevice
                   26.99%  22.200ms         1  22.200ms  22.200ms  22.200ms  cuDevicePrimaryCtxRelease
                    1.50%  1.2344ms         1  1.2344ms  1.2344ms  1.2344ms  cudaMemcpyToSymbol
                    0.45%  370.20us         8  46.275us  9.8000us  157.10us  cudaMemcpy
                    0.19%  155.10us         1  155.10us  155.10us  155.10us  cuLibraryUnload
                    0.17%  136.50us         4  34.125us  3.3000us  123.10us  cudaMalloc
                    0.16%  127.60us         3  42.533us  10.600us  104.50us  cudaLaunchKernel
                    0.08%  67.300us         4  16.825us  4.2000us  47.600us  cudaMemset
                    0.05%  37.100us         3  12.366us  10.100us  15.900us  cudaDeviceSynchronize
                    0.04%  33.300us       114     292ns       0ns  15.400us  cuDeviceGetAttribute
                    0.00%  3.5000us         1  3.5000us  3.5000us  3.5000us  cudaGetDeviceProperties
                    0.00%  2.8000us         3     933ns     100ns  2.4000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 12
==37708== NVPROF is profiling process 37708, command: ./Cuda.exe 12
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 0 ms
[host] Arrays match.

==37708== Profiling application: ./Cuda.exe 12
==37708== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==37708== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.40%  23.264us         4  5.8160us  4.3840us  7.1040us  bfs_kernel(int*, int*, int)
                   28.80%  10.912us         7  1.5580us  1.3760us  2.3040us  [CUDA memcpy DtoH]
                    5.32%  2.0160us         4     504ns     320ns     992ns  [CUDA memcpy HtoD]
                    4.48%  1.6960us         5     339ns     320ns     352ns  [CUDA memset]
      API calls:   69.96%  61.396ms         1  61.396ms  61.396ms  61.396ms  cudaSetDevice
                   26.73%  23.454ms         1  23.454ms  23.454ms  23.454ms  cuDevicePrimaryCtxRelease
                    1.99%  1.7504ms         1  1.7504ms  1.7504ms  1.7504ms  cudaMemcpyToSymbol
                    0.48%  423.20us        10  42.320us  8.4000us  162.90us  cudaMemcpy
                    0.37%  322.50us         1  322.50us  322.50us  322.50us  cuLibraryUnload
                    0.17%  148.30us         4  37.075us  2.7000us  134.90us  cudaMalloc
                    0.10%  92.100us         4  23.025us  7.6000us  63.700us  cudaLaunchKernel
                    0.09%  78.700us         5  15.740us  4.2000us  27.800us  cudaMemset
                    0.05%  48.100us         4  12.025us  10.800us  14.200us  cudaDeviceSynchronize
                    0.04%  32.500us       114     285ns       0ns  14.800us  cuDeviceGetAttribute
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 13
==25080== NVPROF is profiling process 25080, command: ./Cuda.exe 13
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 0 ms
[host] Arrays match.

==25080== Profiling application: ./Cuda.exe 13
==25080== Warning: 36 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==25080== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.50%  17.120us         3  5.7060us  5.6320us  5.8240us  bfs_kernel(int*, int*, int)
                   32.73%  9.9190us         6  1.6530us  1.3120us  2.3350us  [CUDA memcpy DtoH]
                    6.55%  1.9840us         4     496ns     320ns     992ns  [CUDA memcpy HtoD]
                    4.22%  1.2800us         4     320ns     320ns     320ns  [CUDA memset]
      API calls:   72.29%  63.906ms         1  63.906ms  63.906ms  63.906ms  cudaSetDevice
                   25.26%  22.332ms         1  22.332ms  22.332ms  22.332ms  cuDevicePrimaryCtxRelease
                    1.36%  1.2001ms         1  1.2001ms  1.2001ms  1.2001ms  cudaMemcpyToSymbol
                    0.42%  370.90us         9  41.211us  11.900us  157.80us  cudaMemcpy
                    0.22%  198.60us         1  198.60us  198.60us  198.60us  cuLibraryUnload
                    0.17%  148.60us         4  37.150us  2.7000us  134.30us  cudaMalloc
                    0.14%  119.80us         3  39.933us  9.1000us  99.600us  cudaLaunchKernel
                    0.07%  57.700us         4  14.425us  4.3000us  29.200us  cudaMemset
                    0.04%  35.100us         3  11.700us  11.000us  13.000us  cudaDeviceSynchronize
                    0.02%  19.200us       114     168ns       0ns  3.1000us  cuDeviceGetAttribute
                    0.01%  4.8000us         1  4.8000us  4.8000us  4.8000us  cudaGetDeviceProperties
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  2.0000us         3     666ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 14
==6984== NVPROF is profiling process 6984, command: ./Cuda.exe 14
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 1 ms
[host] Arrays match.

==6984== Profiling application: ./Cuda.exe 14
==6984== Warning: 36 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==6984== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.69%  10.336us         2  5.1680us  4.5760us  5.7600us  bfs_kernel(int*, int*, int)
                   27.58%  5.1190us         3  1.7060us  1.3440us  2.3040us  [CUDA memcpy DtoH]
                   11.04%  2.0480us         4     512ns     352ns     992ns  [CUDA memcpy HtoD]
                    5.69%  1.0560us         3     352ns     352ns     352ns  [CUDA memset]
      API calls:   72.81%  65.916ms         1  65.916ms  65.916ms  65.916ms  cudaSetDevice
                   24.61%  22.277ms         1  22.277ms  22.277ms  22.277ms  cuDevicePrimaryCtxRelease
                    1.46%  1.3175ms         1  1.3175ms  1.3175ms  1.3175ms  cudaMemcpyToSymbol
                    0.47%  425.90us         6  70.983us  6.6000us  276.60us  cudaMemcpy
                    0.25%  228.30us         4  57.075us  2.6000us  214.70us  cudaMalloc
                    0.20%  177.50us         1  177.50us  177.50us  177.50us  cuLibraryUnload
                    0.09%  77.100us         2  38.550us  11.500us  65.600us  cudaLaunchKernel
                    0.04%  38.100us         3  12.700us  5.4000us  20.300us  cudaMemset
                    0.04%  32.000us       114     280ns       0ns  15.300us  cuDeviceGetAttribute
                    0.03%  26.800us         2  13.400us  12.700us  14.100us  cudaDeviceSynchronize
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         3     633ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 15
==41728== NVPROF is profiling process 41728, command: ./Cuda.exe 15
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 1 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 0 ms
[host] Arrays match.

==41728== Profiling application: ./Cuda.exe 15
==41728== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==41728== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   66.49%  44.255us         6  7.3750us  5.6960us  8.5120us  bfs_kernel(int*, int*, int)
                   27.07%  18.017us        12  1.5010us  1.3440us  2.3040us  [CUDA memcpy DtoH]
                    3.46%  2.3040us         7     329ns     320ns     352ns  [CUDA memset]
                    2.98%  1.9830us         4     495ns     320ns     991ns  [CUDA memcpy HtoD]
      API calls:   78.30%  93.627ms         1  93.627ms  93.627ms  93.627ms  cudaSetDevice
                   19.74%  23.608ms         1  23.608ms  23.608ms  23.608ms  cuDevicePrimaryCtxRelease
                    1.07%  1.2844ms         1  1.2844ms  1.2844ms  1.2844ms  cudaMemcpyToSymbol
                    0.33%  397.40us        15  26.493us  9.3000us  136.20us  cudaMemcpy
                    0.15%  183.60us         4  45.900us  2.7000us  162.50us  cudaMalloc
                    0.13%  156.50us         1  156.50us  156.50us  156.50us  cuLibraryUnload
                    0.10%  120.90us         6  20.150us  7.0000us  66.800us  cudaLaunchKernel
                    0.07%  82.900us         7  11.842us  3.8000us  25.600us  cudaMemset
                    0.07%  80.200us         6  13.366us  11.000us  14.800us  cudaDeviceSynchronize
                    0.02%  29.000us       114     254ns       0ns  10.100us  cuDeviceGetAttribute
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 16
==41256== NVPROF is profiling process 41256, command: ./Cuda.exe 16
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 1 ms
[host] Arrays match.

==41256== Profiling application: ./Cuda.exe 16
==41256== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==41256== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.90%  4.5120us         1  4.5120us  4.5120us  4.5120us  bfs_kernel(int*, int*, int)
                   27.06%  2.7200us         1  2.7200us  2.7200us  2.7200us  [CUDA memcpy DtoH]
                   21.02%  2.1130us         4     528ns     320ns  1.0880us  [CUDA memcpy HtoD]
                    7.01%     705ns         2     352ns     352ns     353ns  [CUDA memset]
      API calls:   72.37%  79.968ms         1  79.968ms  79.968ms  79.968ms  cudaSetDevice
                   25.36%  28.018ms         1  28.018ms  28.018ms  28.018ms  cuDevicePrimaryCtxRelease
                    1.43%  1.5856ms         1  1.5856ms  1.5856ms  1.5856ms  cudaMemcpyToSymbol
                    0.31%  338.00us         4  84.500us  7.7000us  144.70us  cudaMemcpy
                    0.18%  201.00us         4  50.250us  2.5000us  184.40us  cudaMalloc
                    0.18%  197.10us         1  197.10us  197.10us  197.10us  cuLibraryUnload
                    0.09%  103.80us         1  103.80us  103.80us  103.80us  cudaLaunchKernel
                    0.04%  41.300us         2  20.650us  14.100us  27.200us  cudaMemset
                    0.02%  19.100us       114     167ns       0ns  2.4000us  cuDeviceGetAttribute
                    0.01%  14.500us         1  14.500us  14.500us  14.500us  cudaDeviceSynchronize
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cuModuleGetLoadingMode
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 17
==42968== NVPROF is profiling process 42968, command: ./Cuda.exe 17
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] datasize (25), grid(1, 1), block(512, 1)
[host] device time : 1 ms
[host] Arrays match.

==42968== Profiling application: ./Cuda.exe 17
==42968== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==42968== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   48.06%  5.9200us         1  5.9200us  5.9200us  5.9200us  bfs_kernel(int*, int*, int)
                   29.87%  3.6800us         2  1.8400us  1.3760us  2.3040us  [CUDA memcpy DtoH]
                   16.62%  2.0480us         4     512ns     352ns     992ns  [CUDA memcpy HtoD]
                    5.45%     671ns         2     335ns     320ns     351ns  [CUDA memset]
      API calls:   72.72%  66.599ms         1  66.599ms  66.599ms  66.599ms  cudaSetDevice
                   24.82%  22.733ms         1  22.733ms  22.733ms  22.733ms  cuDevicePrimaryCtxRelease
                    1.63%  1.4944ms         1  1.4944ms  1.4944ms  1.4944ms  cudaMemcpyToSymbol
                    0.35%  320.10us         5  64.020us  9.2000us  205.50us  cudaMemcpy
                    0.17%  151.30us         1  151.30us  151.30us  151.30us  cuLibraryUnload
                    0.16%  147.40us         4  36.850us  2.7000us  134.30us  cudaMalloc
                    0.07%  65.000us         1  65.000us  65.000us  65.000us  cudaLaunchKernel
                    0.04%  32.700us         2  16.350us  12.700us  20.000us  cudaMemset
                    0.02%  20.700us       114     181ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.01%  11.800us         1  11.800us  11.800us  11.800us  cudaDeviceSynchronize
                    0.00%  3.9000us         1  3.9000us  3.9000us  3.9000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  2.2000us         3     733ns       0ns  1.9000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
