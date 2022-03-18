// Adapted from
// https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/solutions/chapter08/cusparse-matrix-matrix.cu
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spmm_csr
//

#include <cuda.h>
#include <cusparse.h>
#include <stdio.h>

// TODO: wrap the
void spmm(){
}

// TODO:
void sddmm(){
}

// TODO: random sparse matric generator

int main(){
    // C = S @ A

    // S
    int S_num_rows = 4, S_num_cols = 4;
    int S_nnz = 9;
    int hS_offsets[] = {0, 3, 4, 7, 9};
    int hS_cols[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    double hS_vals[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // A
    int A_num_rows = S_num_cols, A_num_cols = 3;
    double hA[] = {1.0f,  2.0f,  3.0f,  4.0f,
              5.0f,  6.0f,  7.0f,  8.0f,
              9.0f, 10.0f, 11.0f, 12.0f };
    double *dA;
    // C
    int C_num_rows = S_num_rows, C_num_cols = A_num_cols;
    double hC[4*3] = {0};
    double *dC;

    int *dS_offsets, *dS_cols;
    double *dS_vals;

    // S
    cudaMalloc((void **)&dS_offsets, (S_num_rows + 1) * sizeof(int));
    cudaMalloc((void **)&dS_cols, S_nnz * sizeof(int));
    cudaMalloc((void **)&dS_vals, S_nnz * sizeof(double));
    // A
    cudaMalloc((void **)&dA, A_num_rows * A_num_cols * sizeof(double));
    // C
    cudaMalloc((void **)&dC, C_num_rows * C_num_cols * sizeof(double));

    // S
    cudaMemcpy(dS_offsets, hS_offsets, (S_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dS_cols, hS_cols, S_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dS_vals, hS_vals, S_nnz * sizeof(double), cudaMemcpyHostToDevice);
    // A
    cudaMemcpy(dA, hA, A_num_rows * A_num_cols * sizeof(double), cudaMemcpyHostToDevice);
    // C
    cudaMemcpy(dC, hC, C_num_rows * C_num_cols * sizeof(double), cudaMemcpyHostToDevice);

    // cusparse
    cusparseSpMatDescr_t S_des;
    cusparseDnMatDescr_t A_des, C_des;

    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    cusparseCreateCsr(&S_des, S_num_rows, S_num_cols, S_nnz,
                      dS_offsets, dS_cols, dS_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateDnMat(&A_des, A_num_rows, A_num_cols, A_num_cols, dA, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&C_des, C_num_rows, C_num_cols, C_num_cols, dC, CUDA_R_64F, CUSPARSE_ORDER_ROW);


    double alpha = 1.0, beta = 0.;
    size_t bufsize = 0;
    cusparseSpMM_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, S_des, A_des, &beta, C_des, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &bufsize);

    void *dBuf = NULL;
    cudaMalloc(&dBuf, bufsize);
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, S_des, A_des, &beta, C_des, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            dBuf);

    // Copy back
    cudaMemcpy(hC, dC, C_num_rows * C_num_cols * sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 3; ++j){
            printf("%f ", hC[i * 3 + j]);
        }
        printf("\n");
    }

    return 0;
}
