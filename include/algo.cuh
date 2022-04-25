#pragma once
#include <assert.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "./matrix.cuh"

#define TILE_WIDTH 16

/*********************
Header for struct 
******************/
// forward declaration

struct Algo{
    // SPMM
    void spmm_with_shm(HostSparseMat &, HostDenseMat &, HostDenseMat &);
    void spmm_no_shm(HostSparseMat &, HostDenseMat &, HostDenseMat &);
    void spmm_with_shm_jim(HostSparseMat &, HostDenseMat &, HostDenseMat &, float *gpu_compute_time = nullptr);
    void spmm_with_shm_jim_transpose_first(HostSparseMat &, HostDenseMat &, HostDenseMat &);

    // SDDMM
    void sddmm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_block_over_nnz(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_block_over_nnz_if_same_row_use_shm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_block_over_nnz_but_in_same_row(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_launch_kernel_as_dense_matrix(HostSparseMat &, HostDenseMat &, HostSparseMat &, float *gpu_compute_time = nullptr);
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism
    void sddmm_dynamic_parallelism(HostSparseMat &, HostDenseMat &, HostSparseMat &);

    // SDDMM_SPMM

    // Naively call our fastest two kernels back to back
    void sddmm_spmm_naive_back2back_calls(HostSparseMat &, HostDenseMat &, HostDenseMat &, float *gpu_compute_time = nullptr);

    // block over C, this will cause duplicate works computing S AAT
    void sddmm_spmm_block_over_output(
            HostSparseMat &, HostDenseMat &, HostDenseMat &);
    void sddmm_spmm_block_over_sparse_launch_as_dense_matrix(
            HostSparseMat &, HostDenseMat &, HostDenseMat &, float *gpu_compute_time = nullptr); // block over S
    void sddmm_spmm_block_over_sparse_dynamic_parallelism();

    void ddmm_seq(HostDenseMat &, HostDenseMat &, HostDenseMat &);
    void sddmm_seq(HostSparseMat &, HostDenseMat &, HostSparseMat &);
};



struct CusparseAlgo{
    void spmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A,
        cusparseDnMatDescr_t &C);
    void sddmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A);
    void sddmm_spmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A, 
        cusparseDnMatDescr_t &C);

    // overload
    void spmm(HostSparseMat &S, HostDenseMat &A, HostDenseMat &C, float *gpu_compute_time = nullptr);
    void sddmm(HostSparseMat &S, HostDenseMat &A, HostSparseMat &C, float *gpu_compute_time = nullptr);
    void sddmm_spmm(HostSparseMat &S, HostDenseMat &A, HostDenseMat &C, float *gpu_compute_time = nullptr);
};
