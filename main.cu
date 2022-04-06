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

#define TILE_WIDTH 2

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
};

struct Algo{
    void spmm();
    void sddmm();
    void sddmm_spmm();
    void ddmm_seq(HostDenseMat &, HostDenseMat &, HostDenseMat &);
    void ddmm(HostDenseMat &, HostDenseMat &, HostDenseMat &);
};

struct MatrixGenerator{
    void generate_sparse_csr(int, int);
    void generate_dense(int, int, double**);
};

struct CusparseAlgo{
    void spmm(cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A, cusparseDnMatDescr_t &C);
    void sddmm();
    void sddmm_spmm();
};


// ---------------------------
void Algo::spmm(){
}

void Algo::sddmm(){
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


__global__ void ddmm_kernel(double* A, double* B, double* C, int width) {
    // A, B, C are square
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    int gx_C = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int gy_C = blockIdx.y * TILE_WIDTH + threadIdx.y;
    double value_C = 0.0;

    for(int m = 0; m < width / TILE_WIDTH; m++) {
        int lx_A = threadIdx.x;
        int ly_A = threadIdx.y;
        int gx_A = m * TILE_WIDTH + threadIdx.x;
        int gy_A = gy_C;
        As[ly_A][lx_A] = A[gy_A * width + gx_A];

        int lx_B = threadIdx.x;
        int ly_B = threadIdx.y;
        int gx_B = gx_C;
        int gy_B = m * TILE_WIDTH + threadIdx.y;
        Bs[ly_B][lx_B] = B[gy_B * width + gx_B];
        
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++)
            value_C += As[ly_A][k] * Bs[k][lx_B];

        __syncthreads(); // remove?

    }

    C[gy_C * width + gx_C] = value_C;
}

void Algo::ddmm(HostDenseMat &A, HostDenseMat &B, HostDenseMat &C){
    DeviceDenseMat d_A, d_B, d_C;
    A.to_device(d_A);
    B.to_device(d_B);
    C.to_device(d_C);

    int width = A.num_rows;
    dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    ddmm_kernel<<<dimGrid, dimBlock>>>(d_A.vals, d_B.vals, d_C.vals, width);

    d_C.copy_to_host(C);
}

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

void MatrixGenerator::generate_sparse_csr(int num_rows_, int num_cols_){
}

void MatrixGenerator::generate_dense(int num_rows_, int num_cols_, double** vals){
    *vals = new double[num_rows_ * num_cols_];
    for(int i = 0; i < num_rows_; i++)
        for(int j = 0; j < num_cols_; j++)
            (*vals)[i*num_cols_+j] = ((double)rand()/(double)RAND_MAX);
}

// ==
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
            os << std::right << std::setw(4) << std::setprecision(2) << obj.vals[i*obj.num_cols + j] << "\t";
        }
        os << "\n";
    }
    return os;
}


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

void test_spmm(){
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

    // matrix A
    int A_num_rows = 4, A_num_cols = 4;
    double* A_vals = NULL;
    mg.generate_dense(A_num_rows, A_num_cols, &A_vals);
    HostDenseMat A(A_num_rows, A_num_cols, A_vals);

    // matrix B
    int B_num_rows = 4, B_num_cols = 4;
    double* B_vals = NULL;
    mg.generate_dense(B_num_rows, B_num_cols, &B_vals);
    HostDenseMat B(B_num_rows, B_num_cols, B_vals);

    // matrix C
    int C_num_rows = A_num_rows, C_num_cols = B_num_cols;
    double* C_vals = NULL;
    mg.generate_dense(C_num_rows, C_num_cols, &C_vals);
    HostDenseMat C(C_num_rows, C_num_cols, C_vals);

    alg.ddmm_seq(A, B, C);
    std::cout << C;

    // matrix D
    int D_num_rows = A_num_rows, D_num_cols = B_num_cols;
    double* D_vals = NULL;
    mg.generate_dense(D_num_rows, D_num_cols, &D_vals);
    HostDenseMat D(D_num_rows, D_num_cols, D_vals);

    alg.ddmm(A, B, D);
    std::cout << D;
}

int main(){
    // test_spmm();
    test_ddmm();
    return 0;
}
