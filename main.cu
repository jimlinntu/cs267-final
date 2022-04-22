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


int main(){
    // Fix the random seed so that things are reproducible
    srand(9999);

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

    bm.benchmark_sddmm(sddmm_result);

    std::cout << "====== sddmm benchmark result: =====\n";
    std::cout << sddmm_result;
    std::cout << "====================================\n";

    bm.benchmark_spmm(spmm_result);
    std::cout << "====== spmm benchmark result: =====\n";
    std::cout << spmm_result;
    std::cout << "====================================\n";

    bm.benchmark_sddmm_spmm(sddmm_spmm_result);
    std::cout << "=== sddmm_spmm benchmark result: ===\n";
    std::cout << sddmm_spmm_result;
    std::cout << "====================================\n";

    return 0;
}
