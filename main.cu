// Adapted from
// https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/solutions/chapter08/cusparse-matrix-matrix.cu
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spmm_csr
//

#include "./include/algo.cuh"
#include "./include/matrix.cuh"
#include <cuda.h>
#include <cusparse.h>
#include <stdio.h>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <limits> 
#include <time.h>


void test_cusparse_spmm(){
    // C = S @ A
    int S_num_rows = 4, S_num_cols = 4;
    int S_nnz = 9;
    int hS_offsets[] = {0, 3, 4, 7, 9};
    int hS_cols[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    double hS_vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    int A_num_rows = S_num_cols, A_num_cols = 3;
    double hA[] = {1.0f,  2.0f,  3.0f,  4.0f,
              5.0f,  6.0f,  7.0f,  8.0f,
              9.0f, 10.0f, 11.0f, 12.0f };

    int C_num_rows = S_num_rows, C_num_cols = A_num_cols;
    double hC[4*3] = {0};


    HostSparseMat S(S_num_rows, S_num_cols, S_nnz,
                  hS_offsets, hS_cols, hS_vals);

    HostDenseMat A(A_num_rows, A_num_cols, hA);

    HostDenseMat C(C_num_rows, C_num_cols, hC);


    DeviceSparseMat dS;
    DeviceDenseMat dA, dC;

    S.to_device(dS);
    A.to_device(dA);
    C.to_device(dC);

    // Initialize environment
    {
        cusparseHandle_t handle = NULL;
        assert(cusparseCreate(&handle) == cudaSuccess);

        cusparseSpMatDescr_t S_des;
        cusparseDnMatDescr_t A_des, C_des;

        // Convert them to cusparse descriptors
        dS.get_cusparse_descriptor(S_des);
        dA.get_cusparse_descriptor(A_des);
        dC.get_cusparse_descriptor(C_des);

        CusparseAlgo cualgo;

        // Execute spmm algorithm
        cualgo.spmm(handle, S_des, A_des, C_des);

        // copy back
        dC.copy_to_host(C);

        // Print the result
        std::cout << C;

        assert(cusparseDestroy(handle) == cudaSuccess);
        assert(cusparseDestroySpMat(S_des) == cudaSuccess);
        assert(cusparseDestroyDnMat(A_des) == cudaSuccess);
        assert(cusparseDestroyDnMat(C_des) == cudaSuccess);
    }
}

void test_ddmm() {
    MatrixGenerator mg;
    Algo alg;
    int A_hs[] = {4, 4, 4, 3, 13};
    int A_ws[] = {4, 8, 16, 1, 5};
    int B_hs[] = {4, 8, 16, 1, 5};
    int B_ws[] = {4, 4, 8, 3, 11};

    for(int i = 0; i < 4; i++) {
        std::cout << "Iteration " << i << ":" << std::endl;
        // matrix A
        int A_num_rows = A_hs[i], A_num_cols = A_ws[i];
        double* A_vals = NULL;
        mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
        HostDenseMat A(A_num_rows, A_num_cols, A_vals);

        // matrix B
        int B_num_rows = B_hs[i], B_num_cols = B_ws[i];
        double* B_vals = NULL;
        mg.generate_dense(B_num_rows, B_num_cols, &B_vals);
        HostDenseMat B(B_num_rows, B_num_cols, B_vals);

        // matrix C
        int C_num_rows = A_num_rows, C_num_cols = B_num_cols;
        double* C_vals = NULL;
        mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
        HostDenseMat C(C_num_rows, C_num_cols, C_vals);

        alg.ddmm_seq(A, B, C);
        std::cout << "Sequential DDMM:" << std::endl;
        std::cout << C;
        // matrix D
        int D_num_rows = A_num_rows, D_num_cols = B_num_cols;
        double* D_vals = NULL;
        mg.generate_dense(D_num_rows, D_num_cols, &D_vals);
        HostDenseMat D(D_num_rows, D_num_cols, D_vals);

        alg.ddmm(A, B, D);
        std::cout << "Blocked DDMM:" << std::endl;
        std::cout << D;
        assert(C == D);
    }
}

void test_spmm(int A_h, int A_w, int B_h, int B_w) {
    MatrixGenerator mg;
    Algo alg;
    int start, end;

    int A_num_rows = A_h, A_num_cols = A_w;
    int A_nnz;
    int *A_offsets, *A_cols;
    double* A_vals;

    mg.generate_sparse_csr(A_num_rows, A_num_cols, A_nnz, &A_cols, &A_offsets, &A_vals);
    HostSparseMat A(A_num_rows, A_num_cols, A_nnz, A_cols, A_offsets, A_vals);
    //std::cout << A << std::endl;

    int B_num_rows = B_h, B_num_cols = B_w;
    double* B_vals;
    mg.generate_dense(B_num_rows, B_num_cols, &B_vals);
    HostDenseMat B(B_num_rows, B_num_cols, B_vals);
    //std::cout << B << std::endl;
    
    int C_num_rows = A_h, C_num_cols = B_w;
    double* C_vals;
    mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
    HostDenseMat C(C_num_rows, C_num_cols, C_vals);


    double* A_dense_vals;
    mg.generate_dense(A_num_rows, A_num_cols, &A_dense_vals);
    HostDenseMat A_dense(A_num_rows, A_num_cols, A_dense_vals);
    A.to_dense(A_dense);
    start = clock();
    alg.ddmm_seq(A_dense, B, C);
    end = clock();

    std::cout << "Sequential DDMM takes " << ((float)end - start)/CLOCKS_PER_SEC << " seconds" << std::endl;

    // std::cout << C;
    

    int D_num_rows = A_h, D_num_cols = B_w;
    double* D_vals;
    mg.generate_dense(D_num_rows, D_num_cols, &D_vals);
    HostDenseMat D(D_num_rows, D_num_cols, D_vals);
    start = clock();
    alg.spmm(A, B, D);
    end = clock();
    std::cout << "Blocked SpMM takes " << ((float)end - start)/CLOCKS_PER_SEC << " seconds" << std::endl;

    // std::cout << D;

    assert(C==D);
}

void test_sddmm() {
    MatrixGenerator mg;
    Algo alg;

    int S_num_rows = 3, S_num_cols = 3;
    int S_nnz;
    int *S_offsets, *S_cols;
    double* S_vals;

    mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);
    HostSparseMat S(S_num_rows, S_num_cols, S_nnz, S_offsets, S_cols, S_vals);
    std::cout << S << std::endl;

    int A_num_rows = 3, A_num_cols = 7;
    double* A_vals;
    mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
    HostDenseMat A(A_num_rows, A_num_cols, A_vals);
    std::cout << A << std::endl;

    int C_num_rows = S_num_rows, C_num_cols = S_num_cols; // same shape as S
    int C_nnz = S_nnz;
    int *C_offsets = new int[C_num_rows+1], *C_cols = new int[C_nnz];
    double* C_vals = new double[C_nnz];

    memcpy(C_offsets, S_offsets, (C_num_rows+1) * sizeof(int));
    memcpy(C_cols, S_cols, C_nnz * sizeof(int));
    memcpy(C_vals, S_vals, C_nnz * sizeof(double));

    HostSparseMat C(C_num_rows, C_num_cols, C_nnz, C_offsets, C_cols, C_vals);

    alg.sddmm_seq(S, A, C);

    std::cout << C << std::endl;

    alg.sddmm(S, A, C);

    std::cout << C << std::endl;
}

int main(){
    srand(time(NULL));

    // test_cusparse_spmm();
    test_spmm(1024, 1024, 1024, 1024);
    // test_ddmm();
    // test_sddmm();
    return 0;
}
