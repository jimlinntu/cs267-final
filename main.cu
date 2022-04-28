// Adapted from
// https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/solutions/chapter08/cusparse-matrix-matrix.cu
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spmm_csr
//

#include "./include/algo.cuh"
#include "./include/matrix.cuh"
#include "./include/benchmark.cuh"
#include "./include/checker.cuh"
#include <cuda.h>
#include <cusparse.h>
#include <stdio.h>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <limits> 
#include <time.h>


int main(int argc, char* argv[]){
    int S_h, S_w, A_h, A_w;

    if(argc != 3) {
        S_h = 1024, A_w  = 1024;
        std::cout << "The arguments are incorrect, and it should be:\n";
        std::cout << "./main <S_height> <A_width>\n";
        std::cout << "We set (S_h, A_w) to (" << S_h << " ," << A_w << ") by default\n"; 
        
    } else {
        S_h = atoi(argv[1]), A_w = atoi(argv[2]);
    }
 
    S_w = S_h; // Assume S is a square matrix
    A_h = S_w; // A_h should be equal to S_w

    std::cout << "===== Matrix Dimension =====\n";
    std::cout << "S_height: " << S_h << "\n";
    std::cout << "S_width: " << S_w << "\n";
    std::cout << "A_height: " << A_h << "\n";
    std::cout << "A_width: " << A_w << "\n";

    // Fix the random seed so that things are reproducible
    srand(9999);

    std::cout << "===== Macro Variable Setting =====\n";
    std::cout << "ZERO_RATIO: " << ZERO_RATIO << std::endl;
    std::cout << "TILE_WIDTH: " << TILE_WIDTH << std::endl;

    // Test correctness
    Checker ch;
    ch.check_correctness_sddmm();
    std::cout << "===== PASS sddmm correctness check =====\n";
    ch.check_correctness_spmm();
    std::cout << "===== PASS spmm correctness check =====\n";
    ch.check_correctness_sddmm_spmm();
    std::cout << "===== PASS sddmm_spmm correctness check =====\n";

    // Test speed
    Benchmarker bm;
    BenchmarkResult sddmm_result, spmm_result, sddmm_spmm_result;

    bm.benchmark_sddmm(sddmm_result, S_h, A_w);

    std::cout << "====== sddmm benchmark result: =====\n";
    std::cout << sddmm_result;
    std::cout << "====================================\n";

    bm.benchmark_spmm(spmm_result, S_h, S_w, A_w);
    std::cout << "====== spmm benchmark result: =====\n";
    std::cout << spmm_result;
    std::cout << "====================================\n";

    bm.benchmark_sddmm_spmm(sddmm_spmm_result, S_h, A_w);
    std::cout << "=== sddmm_spmm benchmark result: ===\n";
    std::cout << sddmm_spmm_result;
    std::cout << "====================================\n";

    return 0;
}
