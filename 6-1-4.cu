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
        assert(edge_number == dst_index);
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
int get_move_count_host(int A[5][5], int sr, int sc, int tr, int tc) {
    int dr[4] = { -1, 1, 0, 0 }, dc[4] = { 0, 0, -1, 1 };
    int visited[5][5] = { 0 }, dist[5][5] = { 0 };
    deque<pii> Q;
    Q.push_back({ sr, sc });
    visited[sr][sc] = 1;
    dist[sr][sc] = 0;
    while (!Q.empty()) {
        pii now = Q.front(); Q.pop_front();
        int r = now.first, c = now.second;
        if (r == tr && c == tc) {
            return dist[r][c];
        }

        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            if (in_range(nr, nc) && visited[nr][nc] == 0 && A[nr][nc] != -1) {
                Q.push_back({ nr,nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }

        for (int i = 0; i < 4; i++) {
            int nr = r, nc = c;
            while (1) {
                if (!in_range(nr + dr[i], nc + dc[i])) break;
                if (A[nr + dr[i]][nc + dc[i]] == -1) break;
                nr += dr[i]; nc += dc[i];
                if (A[nr][nc] == 7) break;
            }
            if (visited[nr][nc] == 0) {
                Q.push_back({ nr,nc });
                dist[nr][nc] = dist[r][c] + 1;
                visited[nr][nc] = 1;
            }
        }
    }
    return -1;
}

int get_move_count_device(int start_vertex, int target_vertex) {
    int n = 25;
    int* h_level = (int*)malloc(n * sizeof(int));
    int* h_newVertexVisited = (int*)malloc(sizeof(int));
    int* d_level, * d_newVertexVisited;
    CHECK(cudaMalloc((void**)&d_level, n * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_newVertexVisited, sizeof(int)));
    fill(h_level, h_level + n, INT_MAX);
    h_level[start_vertex] = 0;
    CHECK(cudaMemcpy(d_level, h_level, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_newVertexVisited, 0, sizeof(int)));

    int answer = -1;
    dim3 block(512, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    //printf("[host] datasize (%d), grid(%d, %d), block(%d, %d)\n", n, grid.x, grid.y, block.x, block.y);
    for (int l = 1; ; l++) {
        CHECK(cudaMemset(d_newVertexVisited, 0, sizeof(int)));
        bfs_kernel << <grid, block >> > (d_level, d_newVertexVisited, l);
        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(h_newVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost));
        //if (*h_newVertexVisited == 0) break;
        CHECK(cudaMemcpy(h_level, d_level, n * sizeof(int), cudaMemcpyDeviceToHost));
        if (*h_newVertexVisited == 0) break;
        if (h_level[target_vertex] != INT_MAX) {
            answer = h_level[target_vertex];
            break;
        }
    }
    return answer;
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

    int* h_answer = (int*)malloc(sizeof(int));
    *h_answer = 0;
    int A[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cin >> A[i][j];
        }
    }

    int sr, sc; cin >> sr >> sc;
    pii target[6];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (A[i][j] > 0 && A[i][j] < 7) {
                target[A[i][j] - 1] = { i, j };
            }
        }
    }

    clock_t start = clock();
    int r = sr, c = sc;
    for (int i = 0; i < 6; i++) {
        int nr = target[i].first, nc = target[i].second;
        int ret = get_move_count_host(A, r, c, nr, nc);
        if (ret == -1) {
            *h_answer = -1; break;
        }
        *h_answer += ret;
        r = nr; c = nc;
    }
    int times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] host time : %d ms\n", times);
    //printf("[host] host answer : %d\n", *h_answer);

    /* host - device code */
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

            for (int k = 0; k < 4; k++) {
                int nr = i, nc = j;
                while (1) {
                    if (!in_range(nr + dr[k], nc + dc[k])) break;
                    if (A[nr + dr[k]][nc + dc[k]] == -1) break;
                    nr += dr[k]; nc += dc[k];
                    if (A[nr][nc] == 7) break;
                }
                if (nr != i || nc != j) {
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
    start = clock();
    int* d_answer = (int*)malloc(sizeof(int));
    *d_answer = 0;
    int prev_vertext = get_vertex_number(sr, sc, 5, 5);
    for (int i = 0; i < 6; i++) {
        int nr = target[i].first, nc = target[i].second;
        int current_vertex = get_vertex_number(nr, nc, 5, 5);
        int ret = get_move_count_device(prev_vertext, current_vertex);
        if (ret == -1) {
            *d_answer = -1; break;
        }
        *d_answer += ret;
        prev_vertext = current_vertex;
    }

    times = ((int)clock() - start) / (CLOCKS_PER_SEC / 1000);
    printf("[host] device time : %d ms\n", times);
   // printf("[host] device answer : %d\n", *d_answer);
    checkResult<int>(h_answer, d_answer, 1);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==6072== NVPROF is profiling process 6072, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==6072== Profiling application: ./Cuda.exe 1
==6072== Warning: 28 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==6072== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   72.60%  182.08us        15  12.138us  8.6720us  14.143us  bfs_kernel(int*, int*, int)
                   18.17%  45.567us        30  1.5180us  1.3120us  2.3040us  [CUDA memcpy DtoH]
                    5.75%  14.430us        21     687ns     320ns  2.0480us  [CUDA memset]
                    3.48%  8.7360us         9     970ns     320ns  1.5360us  [CUDA memcpy HtoD]
      API calls:   80.61%  116.38ms         1  116.38ms  116.38ms  116.38ms  cudaSetDevice
                   17.27%  24.935ms         1  24.935ms  24.935ms  24.935ms  cuDevicePrimaryCtxRelease
                    0.75%  1.0813ms         1  1.0813ms  1.0813ms  1.0813ms  cudaMemcpyToSymbol
                    0.57%  819.60us        38  21.568us  4.9000us  99.100us  cudaMemcpy
                    0.23%  327.50us        15  21.833us  14.400us  44.900us  cudaDeviceSynchronize
                    0.21%  303.10us        15  20.206us  7.3000us  143.90us  cudaLaunchKernel
                    0.13%  190.80us        14  13.628us  2.1000us  140.60us  cudaMalloc
                    0.10%  139.40us        21  6.6380us  3.4000us  35.000us  cudaMemset
                    0.09%  136.20us         1  136.20us  136.20us  136.20us  cuLibraryUnload
                    0.02%  30.600us       114     268ns       0ns  3.5000us  cuDeviceGetAttribute
                    0.01%  10.100us         1  10.100us  10.100us  10.100us  cuModuleGetLoadingMode
                    0.00%  6.1000us         1  6.1000us  6.1000us  6.1000us  cudaGetDeviceProperties
                    0.00%  5.0000us         2  2.5000us     100ns  4.9000us  cuDeviceGet
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cuDeviceTotalMem
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceGetName
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==28328== NVPROF is profiling process 28328, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==28328== Profiling application: ./Cuda.exe 2
==28328== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28328== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   70.50%  144.19us        12  12.016us  8.5120us  15.104us  bfs_kernel(int*, int*, int)
                   18.31%  37.442us        24  1.5600us  1.3120us  2.4320us  [CUDA memcpy DtoH]
                    6.02%  12.321us        18     684ns     320ns  1.8560us  [CUDA memset]
                    5.16%  10.560us         9  1.1730us     384ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   70.02%  81.200ms         1  81.200ms  81.200ms  81.200ms  cudaSetDevice
                   26.96%  31.265ms         1  31.265ms  31.265ms  31.265ms  cuDevicePrimaryCtxRelease
                    1.38%  1.6005ms         1  1.6005ms  1.6005ms  1.6005ms  cudaMemcpyToSymbol
                    0.57%  660.50us        32  20.640us  5.0000us  148.10us  cudaMemcpy
                    0.32%  367.10us        12  30.591us  7.2000us  263.70us  cudaLaunchKernel
                    0.25%  289.50us        14  20.678us  2.0000us  207.80us  cudaMalloc
                    0.19%  220.40us        12  18.366us  14.000us  25.700us  cudaDeviceSynchronize
                    0.16%  183.50us         1  183.50us  183.50us  183.50us  cuLibraryUnload
                    0.11%  133.10us        18  7.3940us  3.3000us  43.300us  cudaMemset
                    0.03%  31.300us       114     274ns       0ns  14.100us  cuDeviceGetAttribute
                    0.00%  3.9000us         1  3.9000us  3.9000us  3.9000us  cudaGetDeviceProperties
                    0.00%  2.3000us         3     766ns     100ns  2.0000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 6
==40588== NVPROF is profiling process 40588, command: ./Cuda.exe 6
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==40588== Profiling application: ./Cuda.exe 6
==40588== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==40588== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.29%  136.90us        14  9.7780us  5.4400us  10.976us  bfs_kernel(int*, int*, int)
                   22.21%  43.874us        28  1.5660us  1.3120us  2.3370us  [CUDA memcpy DtoH]
                    5.18%  10.242us        19     539ns     320ns  1.3440us  [CUDA memset]
                    3.32%  6.5600us         8     820ns     352ns  1.7600us  [CUDA memcpy HtoD]
      API calls:   74.35%  77.965ms         1  77.965ms  77.965ms  77.965ms  cudaSetDevice
                   22.70%  23.800ms         1  23.800ms  23.800ms  23.800ms  cuDevicePrimaryCtxRelease
                    1.06%  1.1095ms         1  1.1095ms  1.1095ms  1.1095ms  cudaMemcpyToSymbol
                    0.83%  865.80us        35  24.737us  5.1000us  126.70us  cudaMemcpy
                    0.30%  313.70us        14  22.407us  7.0000us  166.90us  cudaLaunchKernel
                    0.24%  251.30us        14  17.950us  11.500us  26.300us  cudaDeviceSynchronize
                    0.23%  237.30us        12  19.775us  2.2000us  150.50us  cudaMalloc
                    0.14%  150.60us         1  150.60us  150.60us  150.60us  cuLibraryUnload
                    0.13%  138.70us        19  7.3000us  3.3000us  31.000us  cudaMemset
                    0.02%  19.500us       114     171ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.6000us         3     866ns     100ns  2.3000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.5000us         2     750ns       0ns  1.5000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 11
==38604== NVPROF is profiling process 38604, command: ./Cuda.exe 11
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==38604== Profiling application: ./Cuda.exe 11
==38604== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==38604== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.84%  84.416us         8  10.552us  8.6400us  13.664us  bfs_kernel(int*, int*, int)
                   20.28%  24.867us        16  1.5540us  1.3120us  2.4010us  [CUDA memcpy DtoH]
                    5.56%  6.8160us         9     757ns     320ns  1.4400us  [CUDA memcpy HtoD]
                    5.32%  6.5250us        14     466ns     319ns  1.3760us  [CUDA memset]
      API calls:   71.65%  70.364ms         1  70.364ms  70.364ms  70.364ms  cudaSetDevice
                   24.34%  23.901ms         1  23.901ms  23.901ms  23.901ms  cuDevicePrimaryCtxRelease
                    2.28%  2.2426ms         1  2.2426ms  2.2426ms  2.2426ms  cudaMemcpyToSymbol
                    0.77%  756.30us        24  31.512us  6.0000us  279.50us  cudaMemcpy
                    0.23%  225.20us        14  16.085us  2.0000us  173.70us  cudaMalloc
                    0.21%  208.80us         1  208.80us  208.80us  208.80us  cuLibraryUnload
                    0.18%  179.80us         8  22.475us  8.9000us  101.50us  cudaLaunchKernel
                    0.16%  152.60us         8  19.075us  13.700us  26.400us  cudaDeviceSynchronize
                    0.14%  133.70us        14  9.5500us  3.4000us  31.800us  cudaMemset
                    0.03%  33.300us       114     292ns       0ns  13.900us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns       0ns  2.0000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 12
==42448== NVPROF is profiling process 42448, command: ./Cuda.exe 12
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 3 ms
[host] Arrays match.

==42448== Profiling application: ./Cuda.exe 12
==42448== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==42448== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   71.20%  149.60us        13  11.507us  8.8640us  14.112us  bfs_kernel(int*, int*, int)
                   19.57%  41.118us        26  1.5810us  1.4710us  2.3040us  [CUDA memcpy DtoH]
                    5.25%  11.038us        19     580ns     320ns  1.3760us  [CUDA memset]
                    3.98%  8.3520us         9     928ns     320ns  1.3440us  [CUDA memcpy HtoD]
      API calls:   66.74%  59.610ms         1  59.610ms  59.610ms  59.610ms  cudaSetDevice
                   28.43%  25.392ms         1  25.392ms  25.392ms  25.392ms  cuDevicePrimaryCtxRelease
                    1.53%  1.3701ms         1  1.3701ms  1.3701ms  1.3701ms  cudaMemcpyToSymbol
                    1.10%  986.20us        34  29.005us  6.4000us  186.50us  cudaMemcpy
                    0.83%  738.50us        13  56.807us  30.900us  305.60us  cudaLaunchKernel
                    0.69%  616.20us        13  47.400us  37.100us  79.700us  cudaDeviceSynchronize
                    0.24%  215.30us        19  11.331us  3.4000us  88.400us  cudaMemset
                    0.21%  187.00us        14  13.357us  2.1000us  126.50us  cudaMalloc
                    0.19%  172.40us         1  172.40us  172.40us  172.40us  cuLibraryUnload
                    0.02%  19.000us       114     166ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         3     633ns       0ns  1.7000us  cuDeviceGetCount
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 13
==41284== NVPROF is profiling process 41284, command: ./Cuda.exe 13
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==41284== Profiling application: ./Cuda.exe 13
==41284== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==41284== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.73%  67.840us         7  9.6910us  7.5840us  13.440us  bfs_kernel(int*, int*, int)
                   23.07%  23.104us        14  1.6500us  1.3440us  2.4320us  [CUDA memcpy DtoH]
                    4.63%  4.6400us         5     928ns     352ns  1.6950us  [CUDA memcpy HtoD]
                    4.57%  4.5760us         9     508ns     320ns  1.8880us  [CUDA memset]
      API calls:   70.53%  59.736ms         1  59.736ms  59.736ms  59.736ms  cudaSetDevice
                   26.32%  22.290ms         1  22.290ms  22.290ms  22.290ms  cuDevicePrimaryCtxRelease
                    1.36%  1.1501ms         1  1.1501ms  1.1501ms  1.1501ms  cudaMemcpyToSymbol
                    0.70%  595.90us        18  33.105us  6.8000us  151.50us  cudaMemcpy
                    0.43%  361.70us         7  51.671us  10.200us  256.90us  cudaLaunchKernel
                    0.18%  152.70us         1  152.70us  152.70us  152.70us  cuLibraryUnload
                    0.18%  150.30us         6  25.050us  2.7000us  118.80us  cudaMalloc
                    0.17%  141.20us         7  20.171us  13.700us  28.200us  cudaDeviceSynchronize
                    0.09%  72.500us         9  8.0550us  4.1000us  24.100us  cudaMemset
                    0.03%  21.800us       114     191ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.02%  16.400us         1  16.400us  16.400us  16.400us  cuDeviceGetName
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 14
==24180== NVPROF is profiling process 24180, command: ./Cuda.exe 14
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==24180== Profiling application: ./Cuda.exe 14
==24180== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24180== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.39%  131.17us        12  10.930us  8.2880us  14.368us  bfs_kernel(int*, int*, int)
                   20.51%  39.330us        24  1.6380us  1.2800us  2.8800us  [CUDA memcpy DtoH]
                    6.12%  11.745us        18     652ns     320ns  1.8880us  [CUDA memset]
                    4.97%  9.5370us         9  1.0590us     352ns  1.8570us  [CUDA memcpy HtoD]
      API calls:   72.36%  67.855ms         1  67.855ms  67.855ms  67.855ms  cudaSetDevice
                   24.42%  22.898ms         1  22.898ms  22.898ms  22.898ms  cuDevicePrimaryCtxRelease
                    1.14%  1.0689ms         1  1.0689ms  1.0689ms  1.0689ms  cudaMemcpyToSymbol
                    0.85%  798.10us        32  24.940us  6.2000us  162.40us  cudaMemcpy
                    0.32%  300.80us        12  25.066us  7.0000us  189.10us  cudaLaunchKernel
                    0.31%  293.10us        14  20.935us  2.2000us  223.10us  cudaMalloc
                    0.24%  229.60us        12  19.133us  14.700us  34.400us  cudaDeviceSynchronize
                    0.19%  178.30us         1  178.30us  178.30us  178.30us  cuLibraryUnload
                    0.13%  123.70us        18  6.8720us  3.3000us  29.500us  cudaMemset
                    0.02%  19.500us       114     171ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.00%  3.0000us         1  3.0000us  3.0000us  3.0000us  cudaGetDeviceProperties
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cuModuleGetLoadingMode
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cuDeviceTotalMem
                    0.00%  1.8000us         3     600ns       0ns  1.5000us  cuDeviceGetCount
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 15
==18136== NVPROF is profiling process 18136, command: ./Cuda.exe 15
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==18136== Profiling application: ./Cuda.exe 15
==18136== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==18136== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.83%  71.808us         7  10.258us  7.4240us  12.928us  bfs_kernel(int*, int*, int)
                   23.04%  24.033us        14  1.7160us  1.4080us  2.4330us  [CUDA memcpy DtoH]
                    4.94%  5.1520us         9     572ns     320ns  1.5680us  [CUDA memset]
                    3.19%  3.3280us         5     665ns     320ns  1.3120us  [CUDA memcpy HtoD]
      API calls:   70.88%  63.634ms         1  63.634ms  63.634ms  63.634ms  cudaSetDevice
                   25.98%  23.322ms         1  23.322ms  23.322ms  23.322ms  cuDevicePrimaryCtxRelease
                    1.28%  1.1451ms         1  1.1451ms  1.1451ms  1.1451ms  cudaMemcpyToSymbol
                    0.78%  703.80us        18  39.100us  8.1000us  193.30us  cudaMemcpy
                    0.33%  297.30us         7  42.471us  10.100us  197.80us  cudaLaunchKernel
                    0.22%  193.30us         6  32.216us  2.2000us  168.30us  cudaMalloc
                    0.21%  187.40us         1  187.40us  187.40us  187.40us  cuLibraryUnload
                    0.16%  146.30us         7  20.900us  14.100us  28.000us  cudaDeviceSynchronize
                    0.13%  113.80us         9  12.644us  4.0000us  36.900us  cudaMemset
                    0.02%  19.900us       114     174ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 16
==29224== NVPROF is profiling process 29224, command: ./Cuda.exe 16
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==29224== Profiling application: ./Cuda.exe 16
==29224== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==29224== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   71.61%  206.11us        20  10.305us  5.9200us  13.375us  bfs_kernel(int*, int*, int)
                   20.71%  59.615us        40  1.4900us  1.3120us  2.3360us  [CUDA memcpy DtoH]
                    4.65%  13.375us        26     514ns     319ns  1.3440us  [CUDA memset]
                    3.02%  8.7040us         9     967ns     352ns  1.4720us  [CUDA memcpy HtoD]
      API calls:   67.48%  63.384ms         1  63.384ms  63.384ms  63.384ms  cudaSetDevice
                   28.05%  26.351ms         1  26.351ms  26.351ms  26.351ms  cuDevicePrimaryCtxRelease
                    1.98%  1.8623ms         1  1.8623ms  1.8623ms  1.8623ms  cudaMemcpyToSymbol
                    1.14%  1.0670ms        48  22.229us  5.1000us  214.10us  cudaMemcpy
                    0.36%  337.30us        20  16.865us  11.500us  25.400us  cudaDeviceSynchronize
                    0.33%  309.50us        20  15.475us  7.1000us  124.20us  cudaLaunchKernel
                    0.27%  252.40us        14  18.028us  2.1000us  196.40us  cudaMalloc
                    0.20%  187.00us         1  187.00us  187.00us  187.00us  cuLibraryUnload
                    0.16%  149.10us        26  5.7340us  3.4000us  23.000us  cudaMemset
                    0.02%  18.500us       114     162ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cudaGetDeviceProperties
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         3     600ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 17
==36412== NVPROF is profiling process 36412, command: ./Cuda.exe 17
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==36412== Profiling application: ./Cuda.exe 17
==36412== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==36412== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.68%  119.71us        10  11.971us  8.9920us  13.504us  bfs_kernel(int*, int*, int)
                   20.09%  34.524us        20  1.7260us  1.3760us  2.6550us  [CUDA memcpy DtoH]
                    5.20%  8.9290us        16     558ns     320ns  2.2400us  [CUDA memset]
                    5.03%  8.6400us         9     960ns     352ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   68.59%  61.563ms         1  61.563ms  61.563ms  61.563ms  cudaSetDevice
                   28.16%  25.278ms         1  25.278ms  25.278ms  25.278ms  cuDevicePrimaryCtxRelease
                    1.24%  1.1130ms         1  1.1130ms  1.1130ms  1.1130ms  cudaMemcpyToSymbol
                    0.81%  729.10us        28  26.039us  4.9000us  136.90us  cudaMemcpy
                    0.40%  355.00us        10  35.500us  7.4000us  257.50us  cudaLaunchKernel
                    0.24%  219.60us        10  21.960us  15.900us  40.500us  cudaDeviceSynchronize
                    0.20%  178.40us        14  12.742us  1.9000us  128.40us  cudaMalloc
                    0.18%  157.60us         1  157.60us  157.60us  157.60us  cuLibraryUnload
                    0.12%  108.00us        16  6.7500us  3.6000us  19.900us  cudaMemset
                    0.04%  33.800us       114     296ns       0ns  15.000us  cuDeviceGetAttribute
                    0.01%  5.7000us         3  1.9000us     100ns  4.0000us  cuDeviceGetCount
                    0.00%  3.0000us         1  3.0000us  3.0000us  3.0000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.5000us         1  1.5000us  1.5000us  1.5000us  cuDeviceGetLuid
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 18
==19676== NVPROF is profiling process 19676, command: ./Cuda.exe 18
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==19676== Profiling application: ./Cuda.exe 18
==19676== Warning: 27 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==19676== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   70.74%  130.08us        11  11.825us  9.9190us  14.847us  bfs_kernel(int*, int*, int)
                   18.88%  34.720us        22  1.5780us  1.3760us  2.3680us  [CUDA memcpy DtoH]
                    5.29%  9.7280us         9  1.0800us     352ns  1.5040us  [CUDA memcpy HtoD]
                    5.08%  9.3430us        17     549ns     320ns  1.3120us  [CUDA memset]
      API calls:   71.14%  63.018ms         1  63.018ms  63.018ms  63.018ms  cudaSetDevice
                   25.11%  22.248ms         1  22.248ms  22.248ms  22.248ms  cuDevicePrimaryCtxRelease
                    1.50%  1.3278ms         1  1.3278ms  1.3278ms  1.3278ms  cudaMemcpyToSymbol
                    0.89%  792.40us        30  26.413us  5.3000us  217.40us  cudaMemcpy
                    0.42%  373.10us        11  33.918us  7.2000us  281.90us  cudaLaunchKernel
                    0.30%  265.40us        14  18.957us  2.0000us  212.80us  cudaMalloc
                    0.24%  211.80us        11  19.254us  16.200us  33.500us  cudaDeviceSynchronize
                    0.19%  170.80us         1  170.80us  170.80us  170.80us  cuLibraryUnload
                    0.17%  148.90us        17  8.7580us  3.3000us  45.900us  cudaMemset
                    0.02%  20.300us       114     178ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cudaGetDeviceProperties
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceGetName
                    0.00%  1.7000us         3     566ns     100ns  1.4000us  cuDeviceGetCount
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 19
==41976== NVPROF is profiling process 41976, command: ./Cuda.exe 19
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==41976== Profiling application: ./Cuda.exe 19
==41976== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==41976== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.36%  164.80us        14  11.771us  8.2880us  13.472us  bfs_kernel(int*, int*, int)
                   18.59%  41.762us        28  1.4910us  1.3120us  2.3360us  [CUDA memcpy DtoH]
                    4.64%  10.432us        20     521ns     320ns  1.2800us  [CUDA memset]
                    3.41%  7.6500us         9     850ns     320ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   68.01%  59.759ms         1  59.759ms  59.759ms  59.759ms  cudaSetDevice
                   27.93%  24.545ms         1  24.545ms  24.545ms  24.545ms  cuDevicePrimaryCtxRelease
                    1.50%  1.3159ms         1  1.3159ms  1.3159ms  1.3159ms  cudaMemcpyToSymbol
                    1.05%  918.40us        36  25.511us  5.4000us  263.60us  cudaMemcpy
                    0.49%  426.70us        14  30.478us  7.1000us  300.30us  cudaLaunchKernel
                    0.32%  279.10us        14  19.935us  13.800us  35.500us  cudaDeviceSynchronize
                    0.27%  239.20us        14  17.085us  2.1000us  186.60us  cudaMalloc
                    0.19%  170.50us        20  8.5250us  3.5000us  46.400us  cudaMemset
                    0.19%  169.00us         1  169.00us  169.00us  169.00us  cuLibraryUnload
                    0.04%  32.600us       114     285ns       0ns  14.500us  cuDeviceGetAttribute
                    0.00%  3.6000us         3  1.2000us     200ns  3.2000us  cuDeviceGetCount
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cudaGetDeviceProperties
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 20
==11672== NVPROF is profiling process 11672, command: ./Cuda.exe 20
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==11672== Profiling application: ./Cuda.exe 20
==11672== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==11672== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   70.43%  170.27us        12  14.189us  10.975us  28.576us  bfs_kernel(int*, int*, int)
                   15.46%  37.379us        24  1.5570us  1.2800us  2.6240us  [CUDA memcpy DtoH]
                   10.52%  25.440us        18  1.4130us     320ns  16.960us  [CUDA memset]
                    3.59%  8.6730us         9     963ns     320ns  1.4720us  [CUDA memcpy HtoD]
      API calls:   69.99%  63.040ms         1  63.040ms  63.040ms  63.040ms  cudaSetDevice
                   25.89%  23.317ms         1  23.317ms  23.317ms  23.317ms  cuDevicePrimaryCtxRelease
                    1.52%  1.3680ms         1  1.3680ms  1.3680ms  1.3680ms  cudaMemcpyToSymbol
                    1.05%  949.50us        32  29.671us  7.4000us  178.20us  cudaMemcpy
                    0.49%  445.20us        12  37.100us  7.3000us  304.80us  cudaLaunchKernel
                    0.30%  272.90us        12  22.741us  17.000us  40.000us  cudaDeviceSynchronize
                    0.28%  250.20us        14  17.871us  2.0000us  194.90us  cudaMalloc
                    0.23%  206.70us         1  206.70us  206.70us  206.70us  cuLibraryUnload
                    0.21%  189.00us        18  10.500us  3.4000us  40.400us  cudaMemset
                    0.02%  18.500us       114     162ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cudaGetDeviceProperties
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
