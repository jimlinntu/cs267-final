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

void test_correctness_ddmm(int A_h, int A_w, int B_h, int B_w) {
    MatrixGenerator mg;
    Algo alg;

    // matrix A
    int A_num_rows = A_h, A_num_cols = A_w;
    double* A_vals = NULL;
    mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
    HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);

    // matrix B
    int B_num_rows = B_h, B_num_cols = B_w;
    double* B_vals = NULL;
    mg.generate_dense(B_num_rows, B_num_cols, &B_vals);
    HostDenseMat B(B_num_rows, B_num_cols, B_vals, true);

    // matrix C
    int C_num_rows = A_num_rows, C_num_cols = B_num_cols;
    double* C_vals = NULL;
    mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
    HostDenseMat C(C_num_rows, C_num_cols, C_vals, true);

    alg.ddmm_seq(A, B, C);

    // matrix D
    int D_num_rows = A_num_rows, D_num_cols = B_num_cols;
    double* D_vals = NULL;
    mg.generate_dense(D_num_rows, D_num_cols, &D_vals);
    HostDenseMat D(D_num_rows, D_num_cols, D_vals, true);

    alg.ddmm(A, B, D);
    // std::cout << D;

    assert(C == D);
}

void test_correctness_spmm(int A_h, int A_w, int B_h, int B_w) {
    MatrixGenerator mg;
    Algo alg;

    int A_num_rows = A_h, A_num_cols = A_w;
    int A_nnz;
    int *A_offsets, *A_cols;
    double* A_vals;

    mg.generate_sparse_csr(A_num_rows, A_num_cols, A_nnz, &A_offsets, &A_cols, &A_vals);
    HostSparseMat A(A_num_rows, A_num_cols, A_nnz, A_cols, A_offsets, A_vals, true);
    //std::cout << A << std::endl;

    int B_num_rows = B_h, B_num_cols = B_w;
    double* B_vals;
    mg.generate_dense(B_num_rows, B_num_cols, &B_vals);
    HostDenseMat B(B_num_rows, B_num_cols, B_vals, true);
    //std::cout << B << std::endl;
    
    int C_num_rows = A_h, C_num_cols = B_w;
    double* C_vals;
    mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
    HostDenseMat C(C_num_rows, C_num_cols, C_vals, true);


    double* A_dense_vals;
    mg.generate_dense(A_num_rows, A_num_cols, &A_dense_vals);
    HostDenseMat A_dense(A_num_rows, A_num_cols, A_dense_vals, true);
    A.to_dense(A_dense);
    alg.ddmm_seq(A_dense, B, C);
    
    int D_num_rows = A_h, D_num_cols = B_w;
    double* D_vals;
    mg.generate_dense(D_num_rows, D_num_cols, &D_vals);
    HostDenseMat D(D_num_rows, D_num_cols, D_vals, true);
    alg.spmm(A, B, D);

    assert(C==D);
}

void test_correctness_sddmm(int S_h, int S_w, int A_h, int A_w) {
}

void test_correctness_sddmm_multiple_times(){

    const int num_testcases = 100;

    MatrixGenerator mg;
    // Initialize algorithms
    Algo algo;
    CusparseAlgo cualgo;

    for(int i = 0; i < num_testcases; ++i){
        // [100, 10000)
        int l = 100, r = 1000;
        int S_num_rows = l + rand() % (r-l);
        // [100, 10000)
        int A_num_cols = l + rand() % (r-l);

        // Generate a square sparse matrix
        int S_nnz;
        int *S_offsets;
        int *S_cols;
        double *S_vals;
        // Because cusparseSDDMM only support binary sparse matrix
        mg.generate_binary_sparse_csr(S_num_rows, S_num_rows, S_nnz, &S_offsets, &S_cols, &S_vals);

        double *A_vals;
        // Generate a dense matrix
        mg.generate_dense(S_num_rows, A_num_cols, &A_vals);

        // Create the output sparse matrix
        int *C_offsets = new int[S_num_rows+1];
        int *C_cols = new int[S_nnz];
        double *C_vals = new double[S_nnz];
        double *C_vals_cusparse = new double[S_nnz];

        memcpy(C_offsets, S_offsets, (S_num_rows+1) * sizeof(int));
        memcpy(C_cols, S_cols, (S_nnz) * sizeof(int));

        HostSparseMat S(S_num_rows, S_num_rows, S_nnz, S_offsets, S_cols, S_vals, true);
        HostDenseMat A(S_num_rows, A_num_cols, A_vals, true);
        HostSparseMat C(S_num_rows, S_num_rows, S_nnz, C_offsets, C_cols, C_vals, true);
        HostSparseMat C_cusparse(S_num_rows, S_num_rows, S_nnz, C_offsets, C_cols, C_vals_cusparse, false);

        // Run cusparsesddmm as the ground truth
        cualgo.sddmm(S, A, C_cusparse);

        // Test a bunch of functions here:
        algo.sddmm_block_over_nnz_but_in_same_row(S, A, C);
        // Compare cusparsesddmm result to our kernel's result
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_nnz, 0);
        algo.sddmm_seq(S, A, C);
        assert(C == C_cusparse);

        // Clean up
        delete[] C_vals_cusparse;
    }
}

void test_speed_spmm(int S_h, int S_w, int A_h, int A_w) {
    MatrixGenerator mg;
    Algo alg;
    int start, end;

    int S_num_rows = S_h, S_num_cols = S_w;
    int S_nnz;
    int *S_offsets, *S_cols;
    double* S_vals;

    mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);
    HostSparseMat S(S_num_rows, S_num_cols, S_nnz, S_offsets, S_cols, S_vals, true);

    int A_num_rows = A_h, A_num_cols = A_w;
    double* A_vals;
    mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
    HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);

    int C_num_rows = S_h, C_num_cols = A_w;
    double* C_vals;
    mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
    HostDenseMat C(C_num_rows, C_num_cols, C_vals, true);

    start = clock();
    alg.spmm(S, A, C);
    end = clock();
    std::cout << "SpMM takes " << ((float)end - start)/CLOCKS_PER_SEC << " seconds" << std::endl;
}

void test_speed_sddmm(int S_h, int S_w, int A_h, int A_w) {
    MatrixGenerator mg;
    Algo alg;
    int start, end;
    int S_num_rows = S_h, S_num_cols = S_w;
    int S_nnz;
    int *S_offsets, *S_cols;
    double* S_vals;

    mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);
    HostSparseMat S(S_num_rows, S_num_cols, S_nnz, S_offsets, S_cols, S_vals, true);
    // std::cout << S << std::endl;

    int A_num_rows = A_h, A_num_cols = A_w;
    double* A_vals;
    mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
    HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);
    // std::cout << A << std::endl;

    int C_num_rows = S_num_rows, C_num_cols = S_num_cols; // same shape as S
    int C_nnz = S_nnz;
    int *C_offsets = new int[C_num_rows+1], *C_cols = new int[C_nnz];
    double* C_vals = new double[C_nnz];

    memcpy(C_offsets, S_offsets, (C_num_rows+1) * sizeof(int));
    memcpy(C_cols, S_cols, C_nnz * sizeof(int));
    memcpy(C_vals, S_vals, C_nnz * sizeof(double));

    HostSparseMat C(C_num_rows, C_num_cols, C_nnz, C_offsets, C_cols, C_vals, true);

    start = clock();
    alg.sddmm(S, A, C);
    end = clock();
    std::cout << "SDDMM takes " << ((float)end - start)/CLOCKS_PER_SEC << " seconds" << std::endl;
}

void test_speed_cusparse_spmm(int S_h, int S_w, int A_h, int A_w){
    MatrixGenerator mg;

    int S_num_rows = S_h, S_num_cols = S_w;
    int S_nnz, *S_offsets, *S_cols;
    double *S_vals;
    mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);
    HostSparseMat S(S_num_rows, S_num_cols, S_nnz,
                  S_offsets, S_cols, S_vals, true);

    int A_num_rows = A_h, A_num_cols = A_w;
    double *A_vals;
    mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
    HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);

    int C_num_rows = S_h, C_num_cols = A_w;
    double *C_vals;
    mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
    HostDenseMat C(C_num_rows, C_num_cols, C_vals, true);

    int start, end;
    start = clock();
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
        // std::cout << C;

        assert(cusparseDestroy(handle) == cudaSuccess);
        assert(cusparseDestroySpMat(S_des) == cudaSuccess);
        assert(cusparseDestroyDnMat(A_des) == cudaSuccess);
        assert(cusparseDestroyDnMat(C_des) == cudaSuccess);

        end = clock();

        std::cout << "Cuparse SpMM takes " << ((float)end - start)/CLOCKS_PER_SEC << " seconds" << std::endl;
    }
}

// FIXME
/*
void test_speed_cusparse_sddmm(int S_h, int S_w, int A_h, int A_w){
    int start, end;

    MatrixGenerator mg;

    int S_num_rows = S_h, S_num_cols = S_w;
    int S_nnz, *S_offsets, *S_cols;
    double *S_vals;
    mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);
    HostSparseMat S(S_num_rows, S_num_cols, S_nnz,
                  S_offsets, S_cols, S_vals, true);

    int A_num_rows = A_h, A_num_cols = A_w;
    double *A_vals;
    mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
    HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);
    
    
    int B_num_rows = A_w, B_num_cols = A_h;
    double *B_vals;
    mg.generate_dense(B_num_rows, B_num_cols, &B_vals);
    HostDenseMat B(B_num_rows, B_num_cols, B_vals, true);

    start = clock();

    DeviceSparseMat dS;
    DeviceDenseMat dA, dB;

    S.to_device(dS);
    A.to_device(dA);
    B.to_device(dB);

    // Initialize environment
    {
        cusparseHandle_t handle = NULL;
        assert(cusparseCreate(&handle) == cudaSuccess);

        cusparseSpMatDescr_t S_des;
        cusparseDnMatDescr_t A_des, B_des;

        // Convert them to cusparse descriptors
        dS.get_cusparse_descriptor(S_des);
        dA.get_cusparse_col_descriptor(A_des);
        dB.get_cusparse_col_descriptor(B_des);

        CusparseAlgo cualgo;

        // Execute sddmm algorithm
        cualgo.sddmm(handle, S_des, A_des, B_des);

        // copy back
        dS.copy_to_host(S);

        // Print the result
        // std::cout << S;

        assert(cusparseDestroy(handle) == cudaSuccess);
        assert(cusparseDestroySpMat(S_des) == cudaSuccess);
        assert(cusparseDestroyDnMat(A_des) == cudaSuccess);
        assert(cusparseDestroyDnMat(B_des) == cudaSuccess);
        end = clock();

        std::cout << "Cuparse SDDMM takes " << ((float)end - start)/CLOCKS_PER_SEC << " seconds" << std::endl;
    }
}
*/

int main(){
    // Fix the random seed so that things are reproducible
    srand(9999);
    int SIZE = 2048;

    test_correctness_sddmm_multiple_times();

    // test_correctness_ddmm(12, 16, 16, 12);
    // test_correctness_spmm(12, 12, 12, 12);
    /* test_speed_spmm(SIZE, SIZE, SIZE, SIZE); */
    /* test_speed_sddmm(SIZE, SIZE, SIZE, SIZE); */
    // test_speed_cusparse_spmm(SIZE, SIZE, SIZE, SIZE);
    // test_speed_cusparse_sddmm(SIZE, SIZE, SIZE, SIZE);
    return 0;
}
