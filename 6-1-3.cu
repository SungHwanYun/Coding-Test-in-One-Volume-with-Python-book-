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
    pii target[1];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (A[i][j] > 0) {
                target[A[i][j] - 1] = { i, j };
            }
        }
    }

    clock_t start = clock();
    int r = sr, c = sc;
    for (int i = 0; i < 1; i++) {
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
    for (int i = 0; i < 1; i++) {
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
    //printf("[host] device answer : %d\n", *d_answer);
    checkResult<int>(h_answer, d_answer, 1);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==19728== NVPROF is profiling process 19728, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==19728== Profiling application: ./Cuda.exe 1
==19728== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==19728== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.24%  40.895us         3  13.631us  12.992us  14.015us  bfs_kernel(int*, int*, int)
                   18.97%  10.592us         6  1.7650us  1.4080us  2.4640us  [CUDA memcpy DtoH]
                    4.18%  2.3360us         4     584ns     352ns  1.2800us  [CUDA memset]
                    3.61%  2.0160us         4     504ns     320ns  1.0240us  [CUDA memcpy HtoD]
      API calls:   72.16%  66.615ms         1  66.615ms  66.615ms  66.615ms  cudaSetDevice
                   25.41%  23.454ms         1  23.454ms  23.454ms  23.454ms  cuDevicePrimaryCtxRelease
                    1.14%  1.0496ms         1  1.0496ms  1.0496ms  1.0496ms  cudaMemcpyToSymbol
                    0.43%  397.20us         9  44.133us  6.9000us  132.00us  cudaMemcpy
                    0.32%  294.30us         1  294.30us  294.30us  294.30us  cuLibraryUnload
                    0.22%  203.60us         3  67.866us  12.000us  160.50us  cudaLaunchKernel
                    0.17%  153.10us         4  38.275us  3.4000us  134.60us  cudaMalloc
                    0.08%  70.900us         3  23.633us  23.100us  24.100us  cudaDeviceSynchronize
                    0.05%  46.500us         4  11.625us  4.5000us  29.900us  cudaMemset
                    0.02%  19.200us       114     168ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.00%  4.2000us         1  4.2000us  4.2000us  4.2000us  cudaGetDeviceProperties
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         3     600ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==17436== NVPROF is profiling process 17436, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==17436== Profiling application: ./Cuda.exe 2
==17436== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==17436== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   75.79%  40.768us         3  13.589us  12.928us  14.048us  bfs_kernel(int*, int*, int)
                   18.02%  9.6950us         6  1.6150us  1.3760us  2.3360us  [CUDA memcpy DtoH]
                    3.75%  2.0160us         4     504ns     320ns     992ns  [CUDA memcpy HtoD]
                    2.44%  1.3120us         4     328ns     320ns     352ns  [CUDA memset]
      API calls:   71.47%  66.248ms         1  66.248ms  66.248ms  66.248ms  cudaSetDevice
                   25.95%  24.054ms         1  24.054ms  24.054ms  24.054ms  cuDevicePrimaryCtxRelease
                    1.17%  1.0825ms         1  1.0825ms  1.0825ms  1.0825ms  cudaMemcpyToSymbol
                    0.58%  536.00us         9  59.555us  7.4000us  199.70us  cudaMemcpy
                    0.26%  245.10us         1  245.10us  245.10us  245.10us  cuLibraryUnload
                    0.25%  229.70us         3  76.566us  18.500us  180.60us  cudaLaunchKernel
                    0.16%  147.90us         4  36.975us  2.7000us  133.90us  cudaMalloc
                    0.08%  76.000us         3  25.333us  18.800us  30.000us  cudaDeviceSynchronize
                    0.04%  36.600us         4  9.1500us  4.1000us  14.700us  cudaMemset
                    0.02%  19.400us       114     170ns       0ns  2.4000us  cuDeviceGetAttribute
                    0.00%  3.9000us         1  3.9000us  3.9000us  3.9000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.4000us         2     700ns     200ns  1.2000us  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 6
==40916== NVPROF is profiling process 40916, command: ./Cuda.exe 6
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==40916== Profiling application: ./Cuda.exe 6
==40916== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==40916== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.77%  19.746us         2  9.8730us  9.2810us  10.465us  bfs_kernel(int*, int*, int)
                   24.21%  7.6160us         4  1.9040us  1.5040us  2.4320us  [CUDA memcpy DtoH]
                    6.61%  2.0800us         3     693ns     320ns  1.4080us  [CUDA memset]
                    6.41%  2.0160us         4     504ns     320ns     992ns  [CUDA memcpy HtoD]
      API calls:   67.60%  62.214ms         1  62.214ms  62.214ms  62.214ms  cudaSetDevice
                   28.42%  26.153ms         1  26.153ms  26.153ms  26.153ms  cuDevicePrimaryCtxRelease
                    2.24%  2.0599ms         1  2.0599ms  2.0599ms  2.0599ms  cudaMemcpyToSymbol
                    0.57%  527.10us         7  75.300us  11.800us  218.20us  cudaMemcpy
                    0.42%  388.90us         4  97.225us  4.1000us  324.20us  cudaMalloc
                    0.34%  309.30us         2  154.65us  16.800us  292.50us  cudaLaunchKernel
                    0.25%  232.00us         1  232.00us  232.00us  232.00us  cuLibraryUnload
                    0.06%  57.300us         2  28.650us  18.500us  38.800us  cudaDeviceSynchronize
                    0.06%  54.800us         3  18.266us  4.9000us  44.800us  cudaMemset
                    0.02%  18.600us       114     163ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  4.3000us         1  4.3000us  4.3000us  4.3000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  1.8000us         3     600ns     100ns  1.4000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 9
==37180== NVPROF is profiling process 37180, command: ./Cuda.exe 9
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==37180== Profiling application: ./Cuda.exe 9
==37180== Warning: 15 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==37180== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.02%  25.855us         3  8.6180us  6.7830us  11.136us  bfs_kernel(int*, int*, int)
                   23.36%  10.591us         6  1.7650us  1.4720us  2.3680us  [CUDA memcpy DtoH]
                   15.17%  6.8800us         4  1.7200us     320ns  5.8880us  [CUDA memset]
                    4.45%  2.0160us         4     504ns     320ns     992ns  [CUDA memcpy HtoD]
      API calls:   68.04%  56.525ms         1  56.525ms  56.525ms  56.525ms  cudaSetDevice
                   28.66%  23.807ms         1  23.807ms  23.807ms  23.807ms  cuDevicePrimaryCtxRelease
                    1.63%  1.3548ms         1  1.3548ms  1.3548ms  1.3548ms  cudaMemcpyToSymbol
                    0.64%  531.50us         9  59.055us  7.5000us  228.00us  cudaMemcpy
                    0.38%  316.10us         3  105.37us  9.8000us  288.90us  cudaLaunchKernel
                    0.24%  201.70us         1  201.70us  201.70us  201.70us  cuLibraryUnload
                    0.18%  149.70us         4  37.425us  2.9000us  127.60us  cudaMalloc
                    0.09%  78.600us         3  26.200us  16.200us  43.600us  cudaDeviceSynchronize
                    0.09%  76.700us         4  19.175us  5.9000us  33.700us  cudaMemset
                    0.03%  24.000us       114     210ns       0ns  3.6000us  cuDeviceGetAttribute
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaGetDeviceProperties
                    0.00%  2.5000us         3     833ns     100ns  1.5000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  2.2000us         2  1.1000us     100ns  2.1000us  cuDeviceGet
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 10
==12056== NVPROF is profiling process 12056, command: ./Cuda.exe 10
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==12056== Profiling application: ./Cuda.exe 10
==12056== Warning: 10 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==12056== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.22%  15.072us         2  7.5360us  6.7840us  8.2880us  bfs_kernel(int*, int*, int)
                   27.46%  7.2330us         4  1.8080us  1.3440us  2.3360us  [CUDA memcpy DtoH]
                   11.55%  3.0430us         4     760ns     352ns  1.9850us  [CUDA memcpy HtoD]
                    3.77%     992ns         3     330ns     320ns     352ns  [CUDA memset]
      API calls:   68.82%  59.208ms         1  59.208ms  59.208ms  59.208ms  cudaSetDevice
                   27.78%  23.898ms         1  23.898ms  23.898ms  23.898ms  cuDevicePrimaryCtxRelease
                    1.73%  1.4881ms         1  1.4881ms  1.4881ms  1.4881ms  cudaMemcpyToSymbol
                    0.61%  524.00us         7  74.857us  11.900us  248.60us  cudaMemcpy
                    0.40%  341.40us         2  170.70us  19.100us  322.30us  cudaLaunchKernel
                    0.25%  212.10us         1  212.10us  212.10us  212.10us  cuLibraryUnload
                    0.20%  175.10us         4  43.775us  7.0000us  141.10us  cudaMalloc
                    0.10%  84.300us         3  28.100us  10.000us  39.800us  cudaMemset
                    0.07%  57.100us         2  28.550us  14.700us  42.400us  cudaDeviceSynchronize
                    0.04%  35.100us       114     307ns       0ns  11.700us  cuDeviceGetAttribute
                    0.00%  4.3000us         1  4.3000us  4.3000us  4.3000us  cudaGetDeviceProperties
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuModuleGetLoadingMode
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  2.2000us         3     733ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.2000us         2     600ns     100ns  1.1000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 11
==35044== NVPROF is profiling process 35044, command: ./Cuda.exe 11
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==35044== Profiling application: ./Cuda.exe 11
==35044== Warning: 37 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==35044== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   74.68%  37.280us         3  12.426us  11.424us  13.408us  bfs_kernel(int*, int*, int)
                   18.53%  9.2480us         6  1.5410us  1.3440us  2.1760us  [CUDA memcpy DtoH]
                    4.10%  2.0460us         4     511ns     351ns     991ns  [CUDA memcpy HtoD]
                    2.69%  1.3440us         4     336ns     320ns     352ns  [CUDA memset]
      API calls:   70.25%  64.906ms         1  64.906ms  64.906ms  64.906ms  cudaSetDevice
                   26.35%  24.344ms         1  24.344ms  24.344ms  24.344ms  cuDevicePrimaryCtxRelease
                    1.42%  1.3094ms         1  1.3094ms  1.3094ms  1.3094ms  cudaMemcpyToSymbol
                    0.99%  916.60us         1  916.60us  916.60us  916.60us  cuLibraryUnload
                    0.49%  453.90us         9  50.433us  7.9000us  211.30us  cudaMemcpy
                    0.25%  227.30us         4  56.825us  4.2000us  186.90us  cudaMalloc
                    0.10%  92.200us         3  30.733us  7.8000us  74.000us  cudaLaunchKernel
                    0.07%  62.100us         3  20.700us  16.400us  23.600us  cudaDeviceSynchronize
                    0.05%  48.800us         4  12.200us  4.6000us  31.700us  cudaMemset
                    0.02%  19.700us       114     172ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.01%  4.8000us         1  4.8000us  4.8000us  4.8000us  cudaGetDeviceProperties
                    0.00%  3.2000us         1  3.2000us  3.2000us  3.2000us  cuModuleGetLoadingMode
                    0.00%  2.7000us         3     900ns     200ns  2.2000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 12
==14008== NVPROF is profiling process 14008, command: ./Cuda.exe 12
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==14008== Profiling application: ./Cuda.exe 12
==14008== Warning: 2 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==14008== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   69.64%  23.776us         2  11.888us  9.8880us  13.888us  bfs_kernel(int*, int*, int)
                   21.46%  7.3280us         4  1.8320us  1.4080us  2.5600us  [CUDA memcpy DtoH]
                    5.90%  2.0160us         4     504ns     320ns     960ns  [CUDA memcpy HtoD]
                    3.00%  1.0230us         3     341ns     320ns     352ns  [CUDA memset]
      API calls:   70.32%  61.228ms         1  61.228ms  61.228ms  61.228ms  cudaSetDevice
                   26.89%  23.412ms         1  23.412ms  23.412ms  23.412ms  cuDevicePrimaryCtxRelease
                    1.29%  1.1188ms         1  1.1188ms  1.1188ms  1.1188ms  cudaMemcpyToSymbol
                    0.55%  481.60us         7  68.800us  9.2000us  196.60us  cudaMemcpy
                    0.28%  242.00us         1  242.00us  242.00us  242.00us  cuLibraryUnload
                    0.28%  240.50us         2  120.25us  15.700us  224.80us  cudaLaunchKernel
                    0.22%  195.60us         4  48.900us  4.6000us  174.00us  cudaMalloc
                    0.07%  65.200us         2  32.600us  21.600us  43.600us  cudaDeviceSynchronize
                    0.05%  41.100us         3  13.700us  4.9000us  29.000us  cudaMemset
                    0.03%  25.500us       114     223ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.8000us         3     933ns     100ns  2.3000us  cuDeviceGetCount
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuDeviceTotalMem
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         2     900ns     200ns  1.6000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 13
==35092== NVPROF is profiling process 35092, command: ./Cuda.exe 13
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==35092== Profiling application: ./Cuda.exe 13
==35092== Warning: 28 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==35092== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   70.45%  23.264us         2  11.632us  9.5680us  13.696us  bfs_kernel(int*, int*, int)
                   20.35%  6.7200us         4  1.6800us  1.3760us  2.3360us  [CUDA memcpy DtoH]
                    6.10%  2.0160us         4     504ns     320ns     992ns  [CUDA memcpy HtoD]
                    3.10%  1.0240us         3     341ns     320ns     352ns  [CUDA memset]
      API calls:   70.74%  71.320ms         1  71.320ms  71.320ms  71.320ms  cudaSetDevice
                   25.91%  26.118ms         1  26.118ms  26.118ms  26.118ms  cuDevicePrimaryCtxRelease
                    2.12%  2.1378ms         1  2.1378ms  2.1378ms  2.1378ms  cudaMemcpyToSymbol
                    0.38%  383.10us         7  54.728us  10.400us  204.80us  cudaMemcpy
                    0.32%  321.10us         4  80.275us  4.1000us  277.50us  cudaMalloc
                    0.27%  271.70us         1  271.70us  271.70us  271.70us  cuLibraryUnload
                    0.14%  140.30us         2  70.150us  15.500us  124.80us  cudaLaunchKernel
                    0.06%  61.500us         2  30.750us  23.000us  38.500us  cudaDeviceSynchronize
                    0.03%  31.100us         3  10.366us  4.8000us  20.700us  cudaMemset
                    0.02%  19.500us       114     171ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.00%  3.9000us         1  3.9000us  3.9000us  3.9000us  cudaGetDeviceProperties
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.6000us         3     533ns     100ns  1.4000us  cuDeviceGetCount
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 14
==22844== NVPROF is profiling process 22844, command: ./Cuda.exe 14
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==22844== Profiling application: ./Cuda.exe 14
==22844== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==22844== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   70.81%  26.240us         2  13.120us  12.288us  13.952us  bfs_kernel(int*, int*, int)
                   18.39%  6.8160us         4  1.7040us  1.5040us  2.3040us  [CUDA memcpy DtoH]
                    5.52%  2.0470us         4     511ns     352ns     960ns  [CUDA memcpy HtoD]
                    5.27%  1.9520us         3     650ns     320ns  1.3120us  [CUDA memset]
      API calls:   70.60%  61.301ms         1  61.301ms  61.301ms  61.301ms  cudaSetDevice
                   26.62%  23.115ms         1  23.115ms  23.115ms  23.115ms  cuDevicePrimaryCtxRelease
                    1.43%  1.2397ms         1  1.2397ms  1.2397ms  1.2397ms  cudaMemcpyToSymbol
                    0.43%  374.80us         7  53.542us  8.1000us  233.90us  cudaMemcpy
                    0.40%  351.10us         1  351.10us  351.10us  351.10us  cuLibraryUnload
                    0.22%  188.30us         4  47.075us  4.1000us  170.60us  cudaMalloc
                    0.14%  122.00us         2  61.000us  10.200us  111.80us  cudaLaunchKernel
                    0.06%  51.100us         2  25.550us  23.000us  28.100us  cudaDeviceSynchronize
                    0.04%  38.900us         3  12.966us  4.8000us  28.400us  cudaMemset
                    0.02%  18.500us       114     162ns       0ns  3.6000us  cuDeviceGetAttribute
                    0.02%  13.900us         1  13.900us  13.900us  13.900us  cuDeviceGetLuid
                    0.01%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuDeviceTotalMem
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         2     850ns       0ns  1.7000us  cuDeviceGet
                    0.00%  1.6000us         3     533ns     100ns  1.3000us  cuDeviceGetCount
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 19
==26996== NVPROF is profiling process 26996, command: ./Cuda.exe 19
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==26996== Profiling application: ./Cuda.exe 19
==26996== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==26996== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.62%  15.136us         1  15.136us  15.136us  15.136us  bfs_kernel(int*, int*, int)
                   16.67%  3.9040us         2  1.9520us  1.4400us  2.4640us  [CUDA memcpy DtoH]
                    9.70%  2.2720us         2  1.1360us     352ns  1.9200us  [CUDA memset]
                    9.02%  2.1120us         4     528ns     352ns  1.0230us  [CUDA memcpy HtoD]
      API calls:   68.51%  61.357ms         1  61.357ms  61.357ms  61.357ms  cudaSetDevice
                   28.40%  25.430ms         1  25.430ms  25.430ms  25.430ms  cuDevicePrimaryCtxRelease
                    1.79%  1.6000ms         1  1.6000ms  1.6000ms  1.6000ms  cudaMemcpyToSymbol
                    0.42%  376.80us         5  75.360us  8.1000us  203.50us  cudaMemcpy
                    0.29%  260.40us         1  260.40us  260.40us  260.40us  cudaLaunchKernel
                    0.28%  253.10us         1  253.10us  253.10us  253.10us  cuLibraryUnload
                    0.19%  165.80us         4  41.450us  3.6000us  145.00us  cudaMalloc
                    0.05%  47.800us         2  23.900us  5.7000us  42.100us  cudaMemset
                    0.03%  30.900us         1  30.900us  30.900us  30.900us  cudaDeviceSynchronize
                    0.02%  19.700us       114     172ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.01%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  2.4000us         3     800ns     100ns  2.1000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 20
==21028== NVPROF is profiling process 21028, command: ./Cuda.exe 20
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==21028== Profiling application: ./Cuda.exe 20
==21028== Warning: 27 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21028== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.57%  28.576us         3  9.5250us  7.0410us  11.039us  bfs_kernel(int*, int*, int)
                   25.02%  11.071us         6  1.8450us  1.4080us  2.4630us  [CUDA memcpy DtoH]
                    5.78%  2.5590us         4     639ns     320ns  1.5360us  [CUDA memset]
                    4.63%  2.0480us         4     512ns     320ns  1.0240us  [CUDA memcpy HtoD]
      API calls:   69.96%  59.351ms         1  59.351ms  59.351ms  59.351ms  cudaSetDevice
                   27.31%  23.169ms         1  23.169ms  23.169ms  23.169ms  cuDevicePrimaryCtxRelease
                    1.25%  1.0600ms         1  1.0600ms  1.0600ms  1.0600ms  cudaMemcpyToSymbol
                    0.51%  434.30us         9  48.255us  9.5000us  162.10us  cudaMemcpy
                    0.29%  247.40us         1  247.40us  247.40us  247.40us  cuLibraryUnload
                    0.23%  197.30us         3  65.766us  12.200us  143.00us  cudaLaunchKernel
                    0.23%  192.40us         4  48.100us  3.7000us  176.60us  cudaMalloc
                    0.10%  85.400us         3  28.466us  17.400us  35.500us  cudaDeviceSynchronize
                    0.08%  65.900us         4  16.475us  4.1000us  29.400us  cudaMemset
                    0.02%  19.000us       114     166ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.01%  5.1000us         1  5.1000us  5.1000us  5.1000us  cudaGetDeviceProperties
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  2.2000us         3     733ns       0ns  2.0000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
