#include "../include/algo.cuh"

#define TILE_WIDTH 2

/*********************
Function for CusparseAlgo
******************/

void CusparseAlgo::spmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A, cusparseDnMatDescr_t &C){

    double alpha = 1.0, beta = 0.;
    size_t bufsize = 0;
    cusparseSpMM_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, S, A, &beta, C, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &bufsize);
    void *dbuf = NULL;
    assert(cudaMalloc(&dbuf, bufsize) == cudaSuccess);
    assert(cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, S, A, &beta, C, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf) == cudaSuccess);
    assert(cudaFree(dbuf) == cudaSuccess);
}

void CusparseAlgo::sddmm(){
}

void CusparseAlgo::sddmm_spmm(){
}

/*********************
Function for Algo
******************/
// ---------------------------

__global__ void spmm_kernel(double *A_vals, int *A_cols, int *A_offsets, double *B_vals, double *C_vals, int A_h, int A_w, int B_h, int B_w) {
    int gy_C = blockIdx.y * 1 + threadIdx.y, gx_C = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    if(gy_C >= A_h || gx_C >= B_w) return;

    int row_C = gy_C;
    int col_C = gx_C;

    int start_idx = A_offsets[row_C], end_idx = A_offsets[row_C+1];
    double value = 0.0;

    
    for(int i = start_idx; i < end_idx; i++) {
        int col_A = A_cols[i];
        double val_A = A_vals[i];
        //printf("A[%d][%d]=%f", row_C, col_A, val_A);
        //printf("B[%d][%d]=%f", col_A, col_C, B_vals[col_A*B_w + col_C]);
        value += val_A * B_vals[col_A*B_w + col_C];
    }

    // printf("C_vals[%d][%d]=%f\n", row_C, col_C, value);
    C_vals[row_C*B_w + col_C] = value;
}

void Algo::spmm(HostSparseMat &A, HostDenseMat &B, HostDenseMat &C){
    DeviceSparseMat dA;
    DeviceDenseMat dB, dC;
    A.to_device(dA);
    B.to_device(dB);
    C.to_device(dC);

    int A_h = A.num_rows, A_w = A.num_cols, B_h = B.num_rows, B_w = B.num_cols;
    dim3 dimGrid((B_w+TILE_WIDTH-1)/TILE_WIDTH, A_h);
    dim3 dimBlock(TILE_WIDTH, 1);

    spmm_kernel<<<dimGrid, dimBlock>>>(dA.vals, dA.cols, dA.offsets, dB.vals, dC.vals, A_h, A_w, B_h, B_w);

    dC.copy_to_host(C);
}

__global__ void sddmm_kernel(double *S_vals, int *S_cols, int *S_offsets, int S_nnz, double *A_vals, double *C_vals, int A_h, int A_w) {
    int idx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if(idx >= S_nnz) return;
    int col_C = S_cols[idx];
    int row_C;

    // for(int i = 0; i < S_nnz; i++)
    //     printf("S_vals[%d]=%f S_cols[%d]=%d\n", i, S_vals[i], i, S_cols[i]);
    
    // for(int i = 0; i < A_h+1; i++)
    //     printf("S_offsets[%d]=%d\n", i, S_offsets[i]);

    int i = 0;
    for(; i < A_h; i++) // find where the index sits
        if(S_offsets[i] > idx)
            break;
    row_C = i-1;

    // printf("row_C=%d col_C=%d\n", row_C, col_C);

    double value = 0.0;
    for(int i = 0; i < A_w; i++)
        value += A_vals[row_C*A_w+i] * A_vals[col_C*A_w+i]; // A_vals[row_C][i] * At_vals[i][col_C] = A_vals[row_C][i] * A_vals[col_C][i]
    C_vals[idx] = S_vals[idx] * value; // C_vals[idx]
}

void Algo::sddmm(HostSparseMat &S, HostDenseMat &A, HostSparseMat &C){
    DeviceSparseMat dS, dC;
    DeviceDenseMat dA;
    S.to_device(dS);
    A.to_device(dA);
    C.to_device(dC);

    int A_h = A.num_rows, A_w = A.num_cols;
    int nnz = S.nnz;
    dim3 dimGrid((nnz+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH);

    sddmm_kernel<<<dimGrid, dimBlock>>>(dS.vals, dS.cols, dS.offsets, dS.nnz, dA.vals, dC.vals, A_h, A_w);

    dC.copy_to_host(C);
}

void Algo::sddmm_seq(HostSparseMat &S, HostDenseMat &A, HostSparseMat &C){
    for(int i = 0; i < C.num_rows; i++){
        int row_C = i;
        int start_idx = C.offsets[i], end_idx = C.offsets[i+1];
        for(int j = start_idx; j < end_idx; j++) {
            int col_C = C.cols[j];
            double value = 0.0;
            for(int k = 0; k < A.num_cols; k++)
                value += A[row_C*A.num_cols+k] * A[col_C*A.num_cols+k];
            C.vals[j] = value * S.vals[j];
        }
    }
}

void Algo::sddmm_spmm(){
}

void Algo::ddmm_seq(HostDenseMat &A, HostDenseMat &B, HostDenseMat &C){
    int A_num_rows = A.num_rows, A_num_cols = A.num_cols;
    int B_num_rows = B.num_rows, B_num_cols = B.num_cols;
    assert(A_num_cols == B_num_rows);

    for(int i = 0; i < A_num_rows; i++) {
        for(int j = 0; j < B_num_cols; j++) {
            C[i*B_num_cols+j] = 0.0;
            for(int k = 0; k < A_num_cols; k++) {
                C[i*B_num_cols+j] += A[i*A_num_cols+k] * B[k*B_num_cols+j];
            }
        }
    }
}


__global__ void ddmm_kernel(double* A, double* B, double* C, int A_h, int A_w, int B_h, int B_w) {
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    int gx_C = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int gy_C = blockIdx.y * TILE_WIDTH + threadIdx.y;
    double value_C = 0.0;

    for(int m = 0; m < (A_w+TILE_WIDTH-1) / TILE_WIDTH; m++) {
        int lx_A = threadIdx.x;
        int ly_A = threadIdx.y;
        int gx_A = m * TILE_WIDTH + threadIdx.x;
        int gy_A = gy_C;
        if(gy_A < A_h && gx_A < A_w)
            As[ly_A][lx_A] = A[gy_A * A_w + gx_A];
        else // out of range
            As[ly_A][lx_A] = 0.0;

        int lx_B = threadIdx.x;
        int ly_B = threadIdx.y;
        int gx_B = gx_C;
        int gy_B = m * TILE_WIDTH + threadIdx.y;
        if(gy_B < B_h && gx_B < B_w)
            Bs[ly_B][lx_B] = B[gy_B * B_w + gx_B];
        else
            Bs[ly_B][lx_B] = 0.0;
        
        // printf("As[%d][%d]=%f\n", ly_A, lx_A, As[ly_A][lx_A]);
        // printf("Bs[%d][%d]=%f\n", ly_B, lx_B, Bs[ly_A][lx_A]);
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++)
            value_C += As[ly_A][k] * Bs[k][lx_B];

        __syncthreads();

    }

    // printf("C[%d][%d]=%f\n", gy_C, gx_C, value_C);
    if(gy_C < A_h && gx_C < B_w) // make sure in range
        C[gy_C * B_w + gx_C] = value_C;
    
}

void Algo::ddmm(HostDenseMat &A, HostDenseMat &B, HostDenseMat &C){
    DeviceDenseMat d_A, d_B, d_C;
    A.to_device(d_A);
    B.to_device(d_B);
    C.to_device(d_C);

    int A_h = A.num_rows, A_w = A.num_cols, B_h = B.num_rows, B_w = B.num_cols;
    dim3 dimGrid((B_w+TILE_WIDTH-1)/TILE_WIDTH, (A_h+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    ddmm_kernel<<<dimGrid, dimBlock>>>(d_A.vals, d_B.vals, d_C.vals, A_h, A_w, B_h, B_w);

    d_C.copy_to_host(C);
}
