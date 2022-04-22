#include "../include/matrix.cuh"

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

    for(int i = 0; i < num_rows_; i++){
        (*offsets)[i] = vals_cursor;
        for(int j = 0; j < num_cols_; j++) {
            if(tmp_vals[i*num_cols_+j] >= epsilon) {
                (*vals)[vals_cursor++] = tmp_vals[i*num_cols_+j];
                (*cols)[cols_cursor++] = j;
            }
        }
    }
    (*offsets)[num_rows_] = vals_cursor;
    assert(vals_cursor == nnz);

    free(tmp_vals);
}

void MatrixGenerator::generate_binary_sparse_csr(int num_rows_, int num_cols_, int &nnz, int** offsets, int** cols, double** vals) {
    double *tmp_vals = new double[num_rows_ * num_cols_];
    double zero_ratio = 0.7;
    double val;
    nnz = 0;

    for(int i = 0; i < num_rows_; i++)
        for(int j = 0; j < num_cols_; j++) {
            double p = ((double)rand()/(double)RAND_MAX);
            if(p < zero_ratio)
                val = 0.0;
            else{
                val = 1.0;
                nnz++;
            }

            tmp_vals[i * num_cols_ + j] = val;
        }
    
    *vals = new double[nnz];
    *cols = new int[nnz];
    *offsets = new int[num_rows_+1];
    int vals_cursor = 0;
    int cols_cursor = 0;

    for(int i = 0; i < num_rows_; i++){
        (*offsets)[i] = vals_cursor;
        for(int j = 0; j < num_cols_; j++) {
            if(tmp_vals[i*num_cols_+j] > 0.) {
                (*vals)[vals_cursor++] = tmp_vals[i*num_cols_+j];
                (*cols)[cols_cursor++] = j;
            }
        }
    }
    (*offsets)[num_rows_] = vals_cursor;
    assert(vals_cursor == nnz);

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

HostDenseMat::HostDenseMat(int num_rows_, int num_cols_, double* vals_, bool to_delete_)
        :num_rows(num_rows_), num_cols(num_cols_), vals(vals_), to_delete(to_delete_){
}

HostDenseMat::~HostDenseMat(){
    if(!to_delete) return;
    delete[] vals;
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

void DeviceDenseMat::get_cusparse_col_descriptor(cusparseDnMatDescr_t &mat){
    assert(cusparseCreateDnMat(&mat, num_rows, num_cols, num_rows, vals, CUDA_R_64F, CUSPARSE_ORDER_COL) == cudaSuccess);
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
            int *offsets_, int *cols_, double *vals_, bool to_delete_)
    :num_rows(num_rows_), num_cols(num_cols_), nnz(nnz_),
     offsets(offsets_), cols(cols_), vals(vals_), to_delete(to_delete_){
}

HostSparseMat::~HostSparseMat(){
    if(!to_delete) return;

    delete[] offsets;
    delete[] cols;
    delete[] vals;
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

bool HostSparseMat::operator==(const HostSparseMat &r){
    if(num_rows != r.num_rows || num_cols != r.num_cols) return false;
    if(nnz != r.nnz) return false;
    if(offsets[num_rows] != r.offsets[num_rows]) return false;

    for(int i = 0; i < num_rows; ++i){
        if(offsets[i] != r.offsets[i]) return false;
    }
    const double epsilon = 1e-4;
    for(int i = 0; i < nnz; ++i){
        if(cols[i] != r.cols[i]) return false;
        if(fabs(vals[i]-r.vals[i]) > epsilon) return false;
    }
    return true;
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
    free(tmp);
    return os;
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
