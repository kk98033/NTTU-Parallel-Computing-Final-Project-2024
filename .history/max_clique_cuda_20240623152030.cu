#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>  // For timing


__device__ bool is_clique(const bool* adjacency_matrix, const int* subset, int subset_size, int n) {
    for (int i = 0; i < subset_size; ++i) {
        for (int j = i + 1; j < subset_size; ++j) {
            if (!adjacency_matrix[subset[i] * n + subset[j]]) {
                return false;
            }
        }
    }
    return true;
}

__global__ void find_max_clique_kernel(const bool* adjacency_matrix, int* best_clique, int* max_clique_size, int n) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= n) return;

    int local_max_clique_size = 0;
    int local_best_clique[100];  // 假設最大節點數為 100

    // printf("Thread %d: Starting kernel execution\n", thread_id);

    for (int start = thread_id; start < n; start += gridDim.x * blockDim.x) {
        int current_clique[100];  // 假設最大節點數為 100
        int current_clique_size = 0;
        for (int i = start; i < n; ++i) {
            current_clique[current_clique_size] = i;
            if (is_clique(adjacency_matrix, current_clique, current_clique_size + 1, n)) {
                current_clique_size++;
                if (current_clique_size > local_max_clique_size) {
                    local_max_clique_size = current_clique_size;
                    for (int j = 0; j < current_clique_size; ++j) {
                        local_best_clique[j] = current_clique[j];
                    }
                }
            }
        }
    }

    // printf("Thread %d: local_max_clique_size = %d\n", thread_id, local_max_clique_size);

    __syncthreads();

    if (local_max_clique_size > atomicMax(max_clique_size, local_max_clique_size)) {
        for (int j = 0; j < local_max_clique_size; ++j) {
            best_clique[j] = local_best_clique[j];
        }
    }

    // printf("Thread %d: Finished kernel execution\n", thread_id);
}

void find_max_clique(const std::vector<std::vector<bool>>& adjacency_matrix, std::vector<int>& best_clique, int n) {
    bool* d_adjacency_matrix;
    int* d_best_clique;
    int* d_max_clique_size;

    size_t matrix_size = n * n * sizeof(bool);
    size_t clique_size = n * sizeof(int);
    cudaMalloc(&d_adjacency_matrix, matrix_size);
    cudaMalloc(&d_best_clique, clique_size);
    cudaMalloc(&d_max_clique_size, sizeof(int));
    cudaMemset(d_max_clique_size, 0, sizeof(int));

    bool* h_adjacency_matrix = new bool[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_adjacency_matrix[i * n + j] = adjacency_matrix[i][j];
        }
    }

    cudaMemcpy(d_adjacency_matrix, h_adjacency_matrix, matrix_size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    printf("Launching kernel with grid_size=%d, block_size=%d\n", grid_size, block_size);

    // 計時開始
    auto start = std::chrono::high_resolution_clock::now();

    find_max_clique_kernel<<<grid_size, block_size>>>(d_adjacency_matrix, d_best_clique, d_max_clique_size, n);
    cudaError_t err = cudaDeviceSynchronize();  // Ensure all CUDA operations are finished

    // 計時結束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;
    std::cout << "Kernel execution time: " << duration_ms.count() << " ms\n";

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    int h_max_clique_size;
    cudaMemcpy(&h_max_clique_size, d_max_clique_size, sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "Debug: h_max_clique_size = " << h_max_clique_size << "\n";

    best_clique.resize(h_max_clique_size);
    cudaMemcpy(best_clique.data(), d_best_clique, h_max_clique_size * sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "Debug: CUDA best_clique: ";
    // for (int i = 0; i < h_max_clique_size; ++i) {
    //     std::cout << best_clique[i] << " ";
    // }
    // std::cout << "\n";

    delete[] h_adjacency_matrix;
    cudaFree(d_adjacency_matrix);
    cudaFree(d_best_clique);
    cudaFree(d_max_clique_size);
}

int main() {
    std::ifstream file("clique_data.txt");
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    int n;
    file >> n;

    std::vector<std::vector<bool>> adjacency_matrix(n, std::vector<bool>(n, false));
    int u, v;
    while (file >> u >> v) {
        adjacency_matrix[u][v] = true;
        adjacency_matrix[v][u] = true;
    }

    std::vector<int> best_clique;
    find_max_clique(adjacency_matrix, best_clique, n);

    std::cout << "Max Clique Size: " << best_clique.size() << "\n";
    std::cout << "Max Clique: ";
    for (int node : best_clique) {
        std::cout << node << " ";
    }
    std::cout << "\n";

    return 0;
}
