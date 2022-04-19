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
    void spmm(HostSparseMat &, HostDenseMat &, HostDenseMat &);
    void spmm_no_shm(HostSparseMat &, HostDenseMat &, HostDenseMat &);

    // SDDMM
    void sddmm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_block_over_nnz(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_block_over_nnz_if_same_row_use_shm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_block_over_nnz_but_in_same_row(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_launch_kernel_as_dense_matrix(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_dynamic_parallelism(); // https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/

    // SDDMM_SPMM
    void sddmm_spmm();

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
    // overload
    void spmm(HostSparseMat &S, HostDenseMat &A, HostDenseMat &C);
    void sddmm(HostSparseMat &S, HostDenseMat &A, HostSparseMat &C);

    void sddmm_spmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &C,
        cusparseDnMatDescr_t &A, 
        cusparseDnMatDescr_t &B,
        cusparseDnMatDescr_t &D, 
        cusparseDnMatDescr_t &E);
};
