#include "../include/algo.cuh"
#include "../include/matrix.cuh"

struct Checker{
    void check_correctness_sddmm();
    void check_correctness_spmm();
    void check_correctness_sddmm_spmm();
};
