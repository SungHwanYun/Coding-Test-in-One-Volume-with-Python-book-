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
        if (*h_newVertexVisited == 0) break;
        CHECK(cudaMemcpy(h_level, d_level, n * sizeof(int), cudaMemcpyDeviceToHost));
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
            if (A[i][j] > 0) {
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
    CHECK(cudaMalloc(&(GDeep.srcPtrs), G.vertex_number * sizeof(G.srcPtrs[0])));
    CHECK(cudaMalloc(&(GDeep.dst), G.edge_number * sizeof(G.dst[0])));
    CHECK(cudaMemcpy(GDeep.srcPtrs, G.srcPtrs, 
        G.vertex_number*sizeof(G.srcPtrs[0]), cudaMemcpyHostToDevice));
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
    //printf("[host] device answer : %d\n", *d_answer);
    checkResult<int>(h_answer, d_answer, 1);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==36116== NVPROF is profiling process 36116, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==36116== Profiling application: ./Cuda.exe 1
==36116== Warning: 26 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==36116== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.49%  162.56us        20  8.1280us  6.9120us  9.1200us  bfs_kernel(int*, int*, int)
                   25.08%  59.522us        40  1.4880us  1.3120us  2.3050us  [CUDA memcpy DtoH]
                    3.68%  8.7370us        26     336ns     320ns     353ns  [CUDA memset]
                    2.75%  6.5280us         9     725ns     320ns  1.3440us  [CUDA memcpy HtoD]
      API calls:   76.26%  114.39ms         1  114.39ms  114.39ms  114.39ms  cudaSetDevice
                   17.88%  26.822ms         1  26.822ms  26.822ms  26.822ms  cuDevicePrimaryCtxRelease
                    4.03%  6.0518ms         1  6.0518ms  6.0518ms  6.0518ms  cudaMemcpyToSymbol
                    0.80%  1.1971ms        48  24.939us  7.8000us  97.500us  cudaMemcpy
                    0.25%  371.20us        20  18.560us  10.600us  94.500us  cudaLaunchKernel
                    0.23%  346.40us        14  24.742us  3.5000us  268.70us  cudaMalloc
                    0.19%  291.80us        20  14.590us  12.300us  16.200us  cudaDeviceSynchronize
                    0.17%  260.80us         1  260.80us  260.80us  260.80us  cuLibraryUnload
                    0.15%  224.00us        26  8.6150us  3.3000us  38.000us  cudaMemset
                    0.01%  20.300us       114     178ns       0ns  3.1000us  cuDeviceGetAttribute
                    0.00%  6.3000us         1  6.3000us  6.3000us  6.3000us  cudaGetDeviceProperties
                    0.00%  4.2000us         3  1.4000us     100ns  3.7000us  cuDeviceGetCount
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cuModuleGetLoadingMode
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==42628== NVPROF is profiling process 42628, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==42628== Profiling application: ./Cuda.exe 3
==42628== Warning: 37 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==42628== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   66.94%  146.91us        18  8.1610us  7.2320us  9.8880us  bfs_kernel(int*, int*, int)
                   24.60%  53.983us        36  1.4990us  1.3120us  2.3040us  [CUDA memcpy DtoH]
                    5.51%  12.094us        24     503ns     320ns  1.4080us  [CUDA memset]
                    2.95%  6.4630us         9     718ns     352ns  1.3430us  [CUDA memcpy HtoD]
      API calls:   71.09%  74.457ms         1  74.457ms  74.457ms  74.457ms  cudaSetDevice
                   24.91%  26.086ms         1  26.086ms  26.086ms  26.086ms  cuDevicePrimaryCtxRelease
                    1.53%  1.5974ms         1  1.5974ms  1.5974ms  1.5974ms  cudaMemcpyToSymbol
                    1.21%  1.2679ms        44  28.815us  6.0000us  316.40us  cudaMemcpy
                    0.28%  292.10us         1  292.10us  292.10us  292.10us  cuLibraryUnload
                    0.27%  281.90us        18  15.661us  7.2000us  73.300us  cudaLaunchKernel
                    0.25%  259.90us        18  14.438us  12.700us  24.300us  cudaDeviceSynchronize
                    0.23%  241.80us        14  17.271us  2.2000us  181.20us  cudaMalloc
                    0.21%  225.10us        24  9.3790us  3.4000us  51.100us  cudaMemset
                    0.02%  19.500us       114     171ns       0ns  3.6000us  cuDeviceGetAttribute
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuDeviceTotalMem
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         3     600ns       0ns  1.5000us  cuDeviceGetCount
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 4
==9936== NVPROF is profiling process 9936, command: ./Cuda.exe 4
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==9936== Profiling application: ./Cuda.exe 4
==9936== Warning: 15 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==9936== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   54.31%  98.559us        12  8.2130us  5.9840us  9.6320us  bfs_kernel(int*, int*, int)
                   35.66%  64.705us        24  2.6960us  1.3440us  15.072us  [CUDA memcpy DtoH]
                    6.01%  10.912us        18     606ns     320ns  1.4400us  [CUDA memset]
                    4.02%  7.2960us         9     810ns     320ns  1.2800us  [CUDA memcpy HtoD]
      API calls:   66.55%  62.881ms         1  62.881ms  62.881ms  62.881ms  cudaSetDevice
                   29.71%  28.075ms         1  28.075ms  28.075ms  28.075ms  cuDevicePrimaryCtxRelease
                    1.84%  1.7382ms         1  1.7382ms  1.7382ms  1.7382ms  cudaMemcpyToSymbol
                    0.90%  854.70us        32  26.709us  4.6000us  187.60us  cudaMemcpy
                    0.23%  217.80us         1  217.80us  217.80us  217.80us  cuLibraryUnload
                    0.22%  206.00us        12  17.166us  13.800us  29.400us  cudaDeviceSynchronize
                    0.21%  195.00us        14  13.928us  2.0000us  150.70us  cudaMalloc
                    0.17%  158.10us        12  13.175us  6.9000us  64.200us  cudaLaunchKernel
                    0.13%  118.40us        18  6.5770us  3.4000us  20.900us  cudaMemset
                    0.03%  29.200us       114     256ns       0ns  5.8000us  cuDeviceGetAttribute
                    0.00%  4.5000us         1  4.5000us  4.5000us  4.5000us  cudaGetDeviceProperties
                    0.00%  3.0000us         3  1.0000us     100ns  2.6000us  cuDeviceGetCount
                    0.00%  2.9000us         1  2.9000us  2.9000us  2.9000us  cuDeviceTotalMem
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 7
==41128== NVPROF is profiling process 41128, command: ./Cuda.exe 7
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==41128== Profiling application: ./Cuda.exe 7
==41128== Warning: 40 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==41128== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   58.43%  21.632us         4  5.4080us  4.3520us  5.9200us  bfs_kernel(int*, int*, int)
                   29.04%  10.752us         7  1.5360us  1.2800us  2.3370us  [CUDA memcpy DtoH]
                    7.17%  2.6560us         5     531ns     320ns  1.3760us  [CUDA memset]
                    5.36%  1.9850us         4     496ns     320ns     992ns  [CUDA memcpy HtoD]
      API calls:   70.18%  60.855ms         1  60.855ms  60.855ms  60.855ms  cudaSetDevice
                   27.10%  23.504ms         1  23.504ms  23.504ms  23.504ms  cuDevicePrimaryCtxRelease
                    1.37%  1.1875ms         1  1.1875ms  1.1875ms  1.1875ms  cudaMemcpyToSymbol
                    0.52%  451.50us        10  45.150us  9.7000us  211.80us  cudaMemcpy
                    0.23%  196.10us         1  196.10us  196.10us  196.10us  cuLibraryUnload
                    0.23%  195.20us         4  48.800us  2.6000us  181.60us  cudaMalloc
                    0.15%  130.10us         4  32.525us  7.4000us  99.200us  cudaLaunchKernel
                    0.10%  90.700us         5  18.140us  4.3000us  36.900us  cudaMemset
                    0.07%  60.300us         4  15.075us  10.400us  24.100us  cudaDeviceSynchronize
                    0.04%  31.600us       114     277ns       0ns  14.000us  cuDeviceGetAttribute
                    0.01%  4.6000us         1  4.6000us  4.6000us  4.6000us  cudaGetDeviceProperties
                    0.00%  2.8000us         3     933ns     100ns  2.5000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 9
==28652== NVPROF is profiling process 28652, command: ./Cuda.exe 9
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==28652== Profiling application: ./Cuda.exe 9
==28652== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28652== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   53.44%  10.208us         2  5.1040us  4.3530us  5.8550us  bfs_kernel(int*, int*, int)
                   30.65%  5.8560us         3  1.9520us  1.3440us  2.3360us  [CUDA memcpy DtoH]
                   10.39%  1.9840us         4     496ns     320ns     992ns  [CUDA memcpy HtoD]
                    5.52%  1.0550us         3     351ns     351ns     352ns  [CUDA memset]
      API calls:   70.04%  61.683ms         1  61.683ms  61.683ms  61.683ms  cudaSetDevice
                   27.63%  24.336ms         1  24.336ms  24.336ms  24.336ms  cuDevicePrimaryCtxRelease
                    1.28%  1.1257ms         1  1.1257ms  1.1257ms  1.1257ms  cudaMemcpyToSymbol
                    0.39%  339.40us         6  56.566us  9.8000us  192.40us  cudaMemcpy
                    0.22%  194.90us         4  48.725us  2.7000us  177.10us  cudaMalloc
                    0.18%  157.90us         1  157.90us  157.90us  157.90us  cuLibraryUnload
                    0.17%  147.90us         2  73.950us  23.200us  124.70us  cudaLaunchKernel
                    0.04%  32.500us         3  10.833us  5.7000us  19.300us  cudaMemset
                    0.03%  23.800us         2  11.900us  10.800us  13.000us  cudaDeviceSynchronize
                    0.02%  19.600us       114     171ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaGetDeviceProperties
                    0.00%  2.4000us         3     800ns     100ns  2.1000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 10
==17556== NVPROF is profiling process 17556, command: ./Cuda.exe 10
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==17556== Profiling application: ./Cuda.exe 10
==17556== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==17556== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.77%  10.368us         2  5.1840us  4.3840us  5.9840us  bfs_kernel(int*, int*, int)
                   28.23%  5.2480us         3  1.7490us  1.4400us  2.3050us  [CUDA memcpy DtoH]
                   10.67%  1.9840us         4     496ns     320ns     992ns  [CUDA memcpy HtoD]
                    5.34%     992ns         3     330ns     320ns     352ns  [CUDA memset]
      API calls:   70.06%  59.755ms         1  59.755ms  59.755ms  59.755ms  cudaSetDevice
                   26.79%  22.845ms         1  22.845ms  22.845ms  22.845ms  cuDevicePrimaryCtxRelease
                    1.98%  1.6905ms         1  1.6905ms  1.6905ms  1.6905ms  cudaMemcpyToSymbol
                    0.41%  349.70us         4  87.425us  4.3000us  329.30us  cudaMalloc
                    0.35%  302.60us         6  50.433us  10.600us  127.20us  cudaMemcpy
                    0.17%  143.90us         1  143.90us  143.90us  143.90us  cuLibraryUnload
                    0.12%  100.40us         2  50.200us  15.800us  84.600us  cudaLaunchKernel
                    0.06%  47.900us         3  15.966us  6.4000us  23.500us  cudaMemset
                    0.03%  22.900us         2  11.450us  10.500us  12.400us  cudaDeviceSynchronize
                    0.02%  18.500us       114     162ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.01%  4.5000us         1  4.5000us  4.5000us  4.5000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.2000us         2     600ns       0ns  1.2000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 11
==28972== NVPROF is profiling process 28972, command: ./Cuda.exe 11
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 2 ms
[host] Arrays match.

==28972== Profiling application: ./Cuda.exe 11
==28972== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==28972== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.91%  140.19us        20  7.0090us  4.7680us  8.5130us  bfs_kernel(int*, int*, int)
                   27.38%  60.066us        40  1.5010us  1.3120us  2.3040us  [CUDA memcpy DtoH]
                    5.73%  12.575us        26     483ns     320ns  1.3440us  [CUDA memset]
                    2.98%  6.5280us         9     725ns     320ns  1.3440us  [CUDA memcpy HtoD]
      API calls:   67.97%  64.523ms         1  64.523ms  64.523ms  64.523ms  cudaSetDevice
                   27.07%  25.693ms         1  25.693ms  25.693ms  25.693ms  cuDevicePrimaryCtxRelease
                    2.02%  1.9201ms         1  1.9201ms  1.9201ms  1.9201ms  cudaMemcpyToSymbol
                    1.42%  1.3435ms        48  27.989us  7.7000us  279.30us  cudaMemcpy
                    0.42%  397.60us        20  19.880us  9.1000us  108.50us  cudaLaunchKernel
                    0.34%  327.40us        14  23.385us  2.8000us  262.90us  cudaMalloc
                    0.32%  302.30us        20  15.115us  10.800us  26.700us  cudaDeviceSynchronize
                    0.20%  192.20us         1  192.20us  192.20us  192.20us  cuLibraryUnload
                    0.20%  191.20us        26  7.3530us  4.5000us  33.600us  cudaMemset
                    0.02%  19.800us       114     173ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.5000us         3     833ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuModuleGetLoadingMode
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  1.2000us         2     600ns       0ns  1.2000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 12
==5652== NVPROF is profiling process 5652, command: ./Cuda.exe 12
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 1 ms
[host] Arrays match.

==5652== Profiling application: ./Cuda.exe 12
==5652== Warning: 22 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==5652== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.79%  127.55us        20  6.3770us  4.7680us  7.4560us  bfs_kernel(int*, int*, int)
                   28.59%  59.010us        40  1.4750us  1.2800us  2.6560us  [CUDA memcpy DtoH]
                    6.07%  12.540us        26     482ns     319ns  1.4400us  [CUDA memset]
                    3.55%  7.3290us         9     814ns     352ns  1.3120us  [CUDA memcpy HtoD]
      API calls:   69.60%  60.740ms         1  60.740ms  60.740ms  60.740ms  cudaSetDevice
                   26.02%  22.706ms         1  22.706ms  22.706ms  22.706ms  cuDevicePrimaryCtxRelease
                    1.95%  1.6974ms         1  1.6974ms  1.6974ms  1.6974ms  cudaMemcpyToSymbol
                    1.11%  966.30us        48  20.131us  5.1000us  148.80us  cudaMemcpy
                    0.34%  299.10us        14  21.364us  1.9000us  246.80us  cudaMalloc
                    0.29%  255.10us        20  12.755us  6.8000us  98.200us  cudaLaunchKernel
                    0.28%  240.20us        20  12.010us  10.000us  14.200us  cudaDeviceSynchronize
                    0.19%  169.10us         1  169.10us  169.10us  169.10us  cuLibraryUnload
                    0.19%  162.50us        26  6.2500us  3.3000us  29.300us  cudaMemset
                    0.02%  20.000us       114     175ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaGetDeviceProperties
                    0.00%  2.6000us         3     866ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 13
==3928== NVPROF is profiling process 3928, command: ./Cuda.exe 13
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 1 ms
[host] device time : 2 ms
[host] Arrays match.

==3928== Profiling application: ./Cuda.exe 13
==3928== Warning: 37 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==3928== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.09%  130.34us        17  7.6660us  5.9840us  8.9600us  bfs_kernel(int*, int*, int)
                   24.99%  50.817us        34  1.4940us  1.2800us  2.2720us  [CUDA memcpy DtoH]
                    6.83%  13.887us        23     603ns     320ns  1.7280us  [CUDA memset]
                    4.09%  8.3180us         9     924ns     351ns  1.3760us  [CUDA memcpy HtoD]
      API calls:   68.20%  58.103ms         1  58.103ms  58.103ms  58.103ms  cudaSetDevice
                   27.16%  23.141ms         1  23.141ms  23.141ms  23.141ms  cuDevicePrimaryCtxRelease
                    2.15%  1.8328ms         1  1.8328ms  1.8328ms  1.8328ms  cudaMemcpyToSymbol
                    1.08%  923.50us        42  21.988us  5.1000us  151.70us  cudaMemcpy
                    0.34%  293.10us        14  20.935us  2.1000us  238.10us  cudaMalloc
                    0.29%  247.40us        17  14.552us  11.600us  30.600us  cudaDeviceSynchronize
                    0.28%  241.70us        17  14.217us  7.0000us  87.700us  cudaLaunchKernel
                    0.24%  204.40us         1  204.40us  204.40us  204.40us  cuLibraryUnload
                    0.19%  160.00us        23  6.9560us  3.3000us  23.100us  cudaMemset
                    0.04%  32.600us       114     285ns       0ns  15.000us  cuDeviceGetAttribute
                    0.01%  4.6000us         1  4.6000us  4.6000us  4.6000us  cudaGetDeviceProperties
                    0.00%  2.0000us         3     666ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 17
==31116== NVPROF is profiling process 31116, command: ./Cuda.exe 17
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450
[host] host time : 0 ms
[host] device time : 0 ms
[host] Arrays match.

==31116== Profiling application: ./Cuda.exe 17
==31116== Warning: 28 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==31116== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.49%  11.936us         2  5.9680us  4.7360us  7.2000us  bfs_kernel(int*, int*, int)
                   25.84%  5.1840us         3  1.7280us  1.4400us  2.2720us  [CUDA memcpy DtoH]
                    9.73%  1.9520us         4     488ns     320ns     960ns  [CUDA memcpy HtoD]
                    4.95%     993ns         3     331ns     320ns     353ns  [CUDA memset]
      API calls:   71.23%  64.166ms         1  64.166ms  64.166ms  64.166ms  cudaSetDevice
                   25.89%  23.319ms         1  23.319ms  23.319ms  23.319ms  cuDevicePrimaryCtxRelease
                    1.77%  1.5924ms         1  1.5924ms  1.5924ms  1.5924ms  cudaMemcpyToSymbol
                    0.41%  371.80us         6  61.966us  8.5000us  179.90us  cudaMemcpy
                    0.27%  239.30us         1  239.30us  239.30us  239.30us  cuLibraryUnload
                    0.24%  215.10us         4  53.775us  6.3000us  183.90us  cudaMalloc
                    0.10%  93.600us         2  46.800us  10.600us  83.000us  cudaLaunchKernel
                    0.03%  24.200us         3  8.0660us  4.2000us  14.500us  cudaMemset
                    0.03%  24.100us         2  12.050us  10.100us  14.000us  cudaDeviceSynchronize
                    0.02%  19.100us       114     167ns       0ns  2.4000us  cuDeviceGetAttribute
                    0.01%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%     800ns         2     400ns     200ns     600ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
