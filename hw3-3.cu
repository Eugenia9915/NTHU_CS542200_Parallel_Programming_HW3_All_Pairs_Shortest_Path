#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <omp.h>

#define BF 32

const int INF = ((1 << 30) - 1);
void input(char* inFileName);
void output(char* outFileName);

void block_FW();
__global__ void cal_1(int, int, int, int*, int);
__global__ void cal_row(int, int, int, int*, int);
__global__ void cal_col(int, int, int, int*, int);
__global__ void cal_3(int, int, int, int*, int);
int ceil(int a, int b) { return (a + b - 1) / b; }

const int V = 40010;
int n, m;

int device_num = 0;
int* Dist = (int*)malloc(sizeof(int)*V*V);
double IO = 0;
double kernel = 0;
int *d_Dist;
int main(int argc, char* argv[]) {
    cudaGetDeviceCount(&device_num);
    printf("devices count: %d\n", device_num);
    //cudamalloc
    cudaMalloc(&d_Dist, V * V * sizeof(int));

    auto start = std::chrono::steady_clock::now();
    input(argv[1]);
    auto finish = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    IO += duration.count();

    cudaMemcpy(d_Dist,Dist,sizeof(int)*n*n,cudaMemcpyHostToDevice);
    start = std::chrono::steady_clock::now();
    block_FW();
    finish = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    kernel += duration.count();

    cudaMemcpy(Dist,d_Dist,sizeof(int)*n*n,cudaMemcpyDeviceToHost);
    start = std::chrono::steady_clock::now();
    output(argv[2]);
    finish = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    IO += duration.count();

    //cudafree
    cudaFree(d_Dist);
    free(Dist);
    
    printf("I/O time: %f ms\n", IO);
    printf("kernel time: %f ms\n", kernel);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) Dist[i * n + j] = 0;
            else Dist[i * n + j] = INF;
        }
    }
    int pair[12];
    int i = 0;
    while(i + 4 < m){
        fread(pair, sizeof(int), 12, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
        Dist[pair[3] * n + pair[4]] = pair[5];
        Dist[pair[6] * n + pair[7]] = pair[8];
        Dist[pair[9] * n + pair[10]] = pair[11];
        i += 4;
    }
    while(i < m){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
        i ++ ;
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
            fwrite(&Dist[i * n + j], sizeof(int), 1, outfile);
        }
    }
    fclose(outfile);
}

void block_FW() {
    int round = ceil(n, BF);
    dim3 num_threads(BF, BF);
    dim3 num_blocks;
    for (int r = 0; r < round; ++r) {
        /* phase 1 */
        num_blocks.x = 1; 
        num_blocks.y = 1;
        cal_1<<<num_blocks,num_threads>>>(r, r, r, d_Dist, n);
        cudaDeviceSynchronize();
        /* phase 2 */
        if(r > 0){
            num_blocks.x = 1;
            num_blocks.y = r;
            cal_row<<<num_blocks,num_threads>>>(r, r, 0, d_Dist, n); //左
            num_blocks.x = r;
            num_blocks.y = 1;
            cal_col<<<num_blocks,num_threads>>>(r, 0, r, d_Dist, n); //上
         }
        if(round - r - 1 > 0){
            num_blocks.x = round - r - 1;
            num_blocks.y = 1;
            cal_col<<<num_blocks,num_threads>>>(r, r + 1, r, d_Dist, n); //下
            num_blocks.x = 1;
            num_blocks.y = round - r - 1;
            cal_row<<<num_blocks,num_threads>>>(r, r, r + 1, d_Dist, n); //右
        }
        cudaDeviceSynchronize();
        /* phase 3 */
        if(r > 0){
            num_blocks.x = r;
            num_blocks.y = r;
            cal_3<<<num_blocks,num_threads>>>(r, 0, 0, d_Dist, n); //左上
        }
        if(r > 0 && round - r - 1 > 0){
            num_blocks.x = r;
            num_blocks.y = round - r - 1;
            cal_3<<<num_blocks,num_threads>>>(r, 0, r + 1, d_Dist, n); //右上
            num_blocks.x = round - r - 1;
            num_blocks.y = r;
            cal_3<<<num_blocks,num_threads>>>(r, r + 1, 0, d_Dist, n); //左下
        }
        if(round - r - 1 > 0){
            num_blocks.x = round - r - 1;
            num_blocks.y = round - r - 1;
            cal_3<<<num_blocks,num_threads>>>(r, r + 1, r + 1, d_Dist, n); //右下
        }
        cudaDeviceSynchronize();
    }
}

__global__ void cal_1(int Round, int block_start_x, int block_start_y, int* Dist, int n) {
    int x = threadIdx.y;
    int y = threadIdx.x;
    int i = (block_start_x + blockIdx.x) * BF + threadIdx.y;
    int j = (block_start_y + blockIdx.y) * BF + threadIdx.x;
    __shared__ int M[BF][BF];
    M[x][y] = Dist[i*n + j];
    __syncthreads();

    if(i >= n || j >= n) return;

    for (int k = 0; k < BF && Round * BF+ k < n; k++) {
        if (M[x][k] + M[k][y] < M[x][y]) M[x][y] = M[x][k] + M[k][y];
        __syncthreads();
    }
    Dist[i*n + j] = M[x][y];
}

__global__ void cal_row(int Round, int block_start_x, int block_start_y, int* Dist, int n) {
    int x = threadIdx.y;
    int y = threadIdx.x;
    int i = (block_start_x + blockIdx.x) * BF + threadIdx.y;
    int j = (block_start_y + blockIdx.y) * BF + threadIdx.x;
    __shared__ int M[BF][BF];
    __shared__ int M1[BF][BF];
    M[x][y] = Dist[i*n + j];
    M1[x][y] = Dist[i*n + Round * BF + y];
    __syncthreads();

    if(i >= n || j >= n) return;
    
    for (int k = 0; k < BF && Round * BF + k < n; k++) {
        if (M1[x][k] + M[k][y] < M[x][y]) M[x][y] = M1[x][k] + M[k][y];
        __syncthreads();
    }
    Dist[i*n + j] = M[x][y];
}


__global__ void cal_col(int Round, int block_start_x, int block_start_y, int* Dist, int n) {
    int x = threadIdx.y;
    int y = threadIdx.x;
    int i = (block_start_x + blockIdx.x) * BF + threadIdx.y;
    int j = (block_start_y + blockIdx.y) * BF + threadIdx.x;
    __shared__ int M[BF][BF];
    __shared__ int M2[BF][BF];
    M[x][y] = Dist[i*n + j];
    M2[x][y] = Dist[((Round * BF+ x) * n) + j];
    __syncthreads();

    if(i >= n || j >= n) return;

    for (int k = 0; k < BF && Round * BF + k < n; k++) {
        if (M[x][k] + M2[k][y] < M[x][y]) M[x][y] = M[x][k] + M2[k][y];
        __syncthreads();
    }
    Dist[i*n + j] = M[x][y];
}

__global__ void cal_3(int Round, int block_start_x, int block_start_y, int* Dist, int nn) {
    int n = nn;    
    int x = threadIdx.y;
    int y = threadIdx.x;
    int i = (block_start_x + blockIdx.x) * BF + threadIdx.y;
    int j = (block_start_y + blockIdx.y) * BF + threadIdx.x;
    __shared__ int M1[BF][BF];
    __shared__ int M2[BF][BF];
    int c = Dist[i*n + j];
    M1[x][y] = Dist[i*n + Round * BF + y];
    M2[x][y] = Dist[((Round * BF + x) * n) + j];
    __syncthreads();

    if(i >= n || j >= n) return;

    for (int k = 0; k < BF && Round * BF + k < n; k++) {
        if (M1[x][k] + M2[k][y] < c) c = M1[x][k] + M2[k][y];
    }
    Dist[i*n + j] = c;
}


