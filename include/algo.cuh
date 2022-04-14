#include <assert.h>
#include "./matrix.cuh"

/*********************
Header for struct 
******************/
// forward declaration

struct Algo{
    void spmm(HostSparseMat &, HostDenseMat &, HostDenseMat &);
    void sddmm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_shm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_spmm();
    void ddmm_seq(HostDenseMat &, HostDenseMat &, HostDenseMat &);
    void ddmm(HostDenseMat &, HostDenseMat &, HostDenseMat &);
    void sddmm_seq(HostSparseMat &, HostDenseMat &, HostSparseMat &);
};



struct CusparseAlgo{
    void spmm(cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A, cusparseDnMatDescr_t &C);
    void sddmm();
    void sddmm_spmm();
};
