#include <assert.h>
#include "./matrix.cuh"

#define TILE_WIDTH 16
#define SPMMSHM
#define SDDMMSHM

/*********************
Header for struct 
******************/
// forward declaration

struct Algo{
    void spmm(HostSparseMat &, HostDenseMat &, HostDenseMat &);
    void sddmm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_spmm();
    void ddmm_seq(HostDenseMat &, HostDenseMat &, HostDenseMat &);
    void ddmm(HostDenseMat &, HostDenseMat &, HostDenseMat &);
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
        cusparseSpMatDescr_t &C,
        cusparseDnMatDescr_t &A,
        cusparseDnMatDescr_t &B);
    void sddmm_spmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &C,
        cusparseDnMatDescr_t &A, 
        cusparseDnMatDescr_t &B,
        cusparseDnMatDescr_t &D, 
        cusparseDnMatDescr_t &E);
};
