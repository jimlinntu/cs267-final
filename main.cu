// Adapted from
// https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/solutions/chapter08/cusparse-matrix-matrix.cu
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spmm_csr
//

#include <cuda.h>
#include <cusparse.h>
#include <stdio.h>
#include <iomanip>
#include <assert.h>
#include <iostream>
#include <cmath>
#include <limits> 

#define TILE_WIDTH 2

/*********************
Header for struct 
******************/
// forward declaration
struct HostDenseMat;
struct DeviceDenseMat;

struct HostSparseMat;
struct DeviceSparseMat;

struct HostDenseMat{
    int num_rows, num_cols;
    double *vals;
    bool to_delete;
    HostDenseMat(int num_rows_, int num_cols_, double *vals_);
    ~HostDenseMat();
    void to_device(DeviceDenseMat &d);
    friend std::ostream& operator<<(std::ostream &os, const HostDenseMat &obj);
    // overload set
    double & operator [](int i) {return vals[i];}
    // overload get
    double operator [](int i) const {return vals[i];};
    // overload comparison
    bool operator == (const HostDenseMat& m2) {
        if(num_rows != m2.num_rows) return false;
        if(num_cols != m2.num_cols) return false;
        double epsilon = 1e-4; // std::numeric_limits<double>::epsilon();
        for(int i = 0; i < num_rows; i++)
            for(int j = 0; j < num_cols; j++)
                if(std::fabs(vals[i*num_cols+j] - m2.vals[i*num_cols+j]) > epsilon)
                    return false;
        return true;
    }
    bool operator != (const HostDenseMat& m2) {
        return !(*this == m2);
    }
};

struct DeviceDenseMat{
    int num_rows, num_cols;
    double *vals;
    DeviceDenseMat() = default;
    ~DeviceDenseMat();

    void get_cusparse_descriptor(cusparseDnMatDescr_t &mat);
    void copy_to_host(HostDenseMat &h);
    // overload set
    __device__ double & operator [](int i) {return vals[i];}
    // overload get
    __device__ double operator [](int i) const {return vals[i];};
};

struct HostSparseMat{
    int num_rows, num_cols;
    int nnz;
    int *offsets;
    int *cols;
    double *vals;
    bool to_delete;
    HostSparseMat(
            int num_rows_, int num_cols_, int nnz_,
            int *offsets_, int *cols_, double *vals_);
    ~HostSparseMat();
    void to_device(DeviceSparseMat &d);
    void to_dense(HostDenseMat &mat);
};

struct DeviceSparseMat{
    int num_rows, num_cols;
    int nnz;
    int *offsets;
    int *cols;
    double *vals;

    DeviceSparseMat() = default;
    DeviceSparseMat(int num_rows_, int num_cols_, int nnz_,
            int *offsets_, int *cols_, double *vals_);
    ~DeviceSparseMat();
    void get_cusparse_descriptor(cusparseSpMatDescr_t &mat);
    void copy_to_host(HostSparseMat &h);
};

struct Algo{
    void spmm(HostSparseMat &, HostDenseMat &, HostDenseMat &);
    void sddmm(HostSparseMat &, HostDenseMat &, HostSparseMat &);
    void sddmm_spmm();
    void ddmm_seq(HostDenseMat &, HostDenseMat &, HostDenseMat &);
    void ddmm(HostDenseMat &, HostDenseMat &, HostDenseMat &);
    void sddmm_seq(HostSparseMat &, HostDenseMat &, HostSparseMat &);
};

struct MatrixGenerator{
    void generate_sparse_csr(int, int, int&, int**, int**, double**);
    void generate_dense(int, int, double**);
};

struct CusparseAlgo{
    void spmm(cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A, cusparseDnMatDescr_t &C);
    void sddmm();
    void sddmm_spmm();
};


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
Function for MatrixGenerator
******************/

void MatrixGenerator::generate_sparse_csr(int num_rows_, int num_cols_, int &nnz, int** offsets, int** cols, double** vals) {
    double *tmp_vals = new double[num_rows_ * num_cols_];
    double epsilon = 1e-4;
    double zero_ratio = 0.7;
    double val;
    nnz = 0;

    for(int i = 0; i < num_rows_; i++)
        for(int j = 0; j < num_cols_; j++) {
            double p = ((double)rand()/(double)RAND_MAX);
            if(p < zero_ratio)
                val = 0.0;
            else
                val = ((double)rand()/(double)RAND_MAX) + epsilon;

            if(val >= epsilon)
                nnz += 1;

            tmp_vals[i * num_cols_ + j] = val;
        }
    
    *vals = new double[nnz];
    *cols = new int[nnz];
    *offsets = new int[num_rows_+1];
    int vals_cursor = 0;
    int cols_cursor = 0;

    // printf("nnz=%d nr=%d nc=%d\n", nnz, num_rows_, num_cols_);

    for(int i = 0; i < num_rows_; i++){
        (*offsets)[i] = vals_cursor;
        for(int j = 0; j < num_cols_; j++) {
            // printf("i=%d j=%d\n", i, j);
            if(tmp_vals[i*num_cols_+j] > epsilon) {
                // printf("i=%d j=%d vals_cursor=%d\n", i, j, vals_cursor);
                (*vals)[vals_cursor++] = tmp_vals[i*num_cols_+j];
                (*cols)[cols_cursor++] = j;
            }
        }
    }
    (*offsets)[num_rows_] = vals_cursor;

    free(tmp_vals);
}

void MatrixGenerator::generate_dense(int num_rows_, int num_cols_, double** vals){
    *vals = new double[num_rows_ * num_cols_];
    for(int i = 0; i < num_rows_; i++)
        for(int j = 0; j < num_cols_; j++)
            (*vals)[i*num_cols_+j] = ((double)rand()/(double)RAND_MAX);
}


/*********************
Function for HostDenseMat
******************/

HostDenseMat::HostDenseMat(int num_rows_, int num_cols_, double* vals_)
        :num_rows(num_rows_), num_cols(num_cols_), vals(vals_), to_delete(false){
}

HostDenseMat::~HostDenseMat(){
    if(!to_delete) return;
    delete vals;
}

void HostDenseMat::to_device(DeviceDenseMat &d){
    d.num_rows = num_rows;
    d.num_cols = num_cols;

    assert(cudaMalloc(&d.vals, num_rows * num_cols * sizeof(double)) == cudaSuccess);
    assert(cudaMemcpy(d.vals, vals, num_rows * num_cols * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
}

std::ostream& operator<<(std::ostream &os, const HostDenseMat &obj){
    for(int i = 0; i < obj.num_rows; ++i){
        for(int j = 0; j < obj.num_cols; ++j){
            os << std::right << std::setw(6) << std::setprecision(4) << obj.vals[i*obj.num_cols + j] << "\t";
        }
        os << "\n";
    }
    return os;
}

/*********************
Function for DeviceDenseMat
******************/

DeviceDenseMat::~DeviceDenseMat(){
    assert(cudaFree(vals) == cudaSuccess);
}
void DeviceDenseMat::get_cusparse_descriptor(cusparseDnMatDescr_t &mat){
    assert(cusparseCreateDnMat(&mat, num_rows, num_cols, num_cols, vals, CUDA_R_64F, CUSPARSE_ORDER_ROW) == cudaSuccess);
}
void DeviceDenseMat::copy_to_host(HostDenseMat &h){
    assert(h.num_rows == num_rows);
    assert(h.num_cols == num_cols);
    assert(cudaMemcpy(h.vals, vals, num_rows * num_cols * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
}

/*********************
Function for HostSparseMat
******************/

HostSparseMat::HostSparseMat(
            int num_rows_, int num_cols_, int nnz_,
            int *offsets_, int *cols_, double *vals_)
    :num_rows(num_rows_), num_cols(num_cols_), nnz(nnz_),
     offsets(offsets_), cols(cols_), vals(vals_), to_delete(false){
}

HostSparseMat::~HostSparseMat(){
    if(!to_delete) return;

    delete offsets;
    delete cols;
    delete vals;
}

void HostSparseMat::to_dense(HostDenseMat &mat){
    for(int i = 0; i < num_rows; i++)
        for(int j = 0; j < num_cols; j++)
            mat.vals[i*num_cols+j] = 0.0;

    for(int i = 0; i < num_rows; i++) {
        int start_idx = offsets[i];
        int end_idx = offsets[i+1];
        for(int j = start_idx; j < end_idx; j++) {
            int col = cols[j];
            mat.vals[i*num_cols+col] = vals[j];
        }
    }
}

void HostSparseMat::to_device(DeviceSparseMat &d){
    d.num_rows = num_rows;
    d.num_cols = num_cols;
    d.nnz = nnz;

    // malloc
    assert(cudaMalloc(&d.offsets, (num_rows+1) * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d.cols, nnz * sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d.vals, nnz * sizeof(double)) == cudaSuccess);

    // copy
    assert(cudaMemcpy(d.offsets, offsets, (num_rows+1) * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d.cols, cols, nnz * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d.vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
}

std::ostream& operator<<(std::ostream &os, const HostSparseMat &obj){
    double* tmp = new double[obj.num_rows * obj.num_cols];

    for(int i = 0; i < obj.num_rows; ++i)
        for(int j = 0; j < obj.num_cols; ++j)
            tmp[i*obj.num_cols + j] = 0;

    for(int i = 0; i < obj.num_rows; i++) {
        int start_idx = obj.offsets[i];
        int end_idx = obj.offsets[i+1];
        for(int j = start_idx; j < end_idx; j++) {
            int col = obj.cols[j];
            tmp[i*obj.num_cols+col] = obj.vals[j];
        }
    }

    for(int i = 0; i < obj.num_rows; ++i){
        for(int j = 0; j < obj.num_cols; ++j){
            os << std::right << std::setw(6) << std::setprecision(4) << tmp[i*obj.num_cols + j] << "\t";
        }
        os << "\n";
    }
    return os;
    free(tmp);
}

/*********************
Function for DeviceSparseMat
******************/

DeviceSparseMat::DeviceSparseMat(
        int num_rows_, int num_cols_, int nnz_,
        int *offsets_, int *cols_, double *vals_)
    :num_rows(num_rows_), num_cols(num_cols_), nnz(nnz_),
     offsets(offsets_), cols(cols_), vals(vals_){
}

DeviceSparseMat::~DeviceSparseMat(){
    assert(cudaFree(offsets) == cudaSuccess);
    assert(cudaFree(cols) == cudaSuccess);
    assert(cudaFree(vals) == cudaSuccess);
}

void DeviceSparseMat::get_cusparse_descriptor(
    cusparseSpMatDescr_t &mat){

    cusparseCreateCsr(&mat, num_rows, num_cols, nnz,
                      offsets, cols, vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
}

void DeviceSparseMat::copy_to_host(HostSparseMat &h){
    assert(h.num_rows == num_rows);
    assert(h.num_cols == num_cols);
    // suppose nnz does not change
    assert(cudaMemcpy(h.vals, vals, nnz * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
}
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

void test_spmm() {
    MatrixGenerator mg;
    Algo alg;


    int A_hs[] = {4, 4, 4, 3, 13};
    int A_ws[] = {4, 8, 16, 1, 5};
    int B_hs[] = {4, 8, 16, 1, 5};
    int B_ws[] = {4, 4, 8, 3, 11};

    for(int i = 0; i < 5; i++){
        std::cout << "Iteration " << i << ":" << std::endl;
        int A_num_rows = A_hs[i], A_num_cols = A_ws[i];
        int A_nnz;
        int *A_offsets, *A_cols;
        double* A_vals;

        mg.generate_sparse_csr(A_num_rows, A_num_cols, A_nnz, &A_cols, &A_offsets, &A_vals);
        HostSparseMat A(A_num_rows, A_num_cols, A_nnz, A_cols, A_offsets, A_vals);
        //std::cout << A << std::endl;

        int B_num_rows = B_hs[i], B_num_cols = B_ws[i];
        double* B_vals;
        mg.generate_dense(B_num_rows, B_num_cols, &B_vals);
        HostDenseMat B(B_num_rows, B_num_cols, B_vals);
        //std::cout << B << std::endl;
        
        int C_num_rows = A_hs[i], C_num_cols = B_ws[i];
        double* C_vals;
        mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
        HostDenseMat C(C_num_rows, C_num_cols, C_vals);


        double* A_dense_vals;
        mg.generate_dense(A_num_rows, A_num_cols, &A_dense_vals);
        HostDenseMat A_dense(A_num_rows, A_num_cols, A_dense_vals);
        A.to_dense(A_dense);

        alg.ddmm_seq(A_dense, B, C);
        std::cout << "Sequential DDMM:" << std::endl;

        std::cout << C;

        int D_num_rows = A_hs[i], D_num_cols = B_ws[i];
        double* D_vals;
        mg.generate_dense(D_num_rows, D_num_cols, &D_vals);
        HostDenseMat D(D_num_rows, D_num_cols, D_vals);
        alg.spmm(A, B, D);
        std::cout << "Blocked SpMM:" << std::endl;

        std::cout << D;

        assert(C==D);
    }
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

    // test_spmm();
    // test_ddmm();
    test_sddmm();
    return 0;
}
