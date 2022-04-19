#include "../include/algo.cuh"

#define MIN(x, y) (((x) < (y))? (x):(y))

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

void CusparseAlgo::sddmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &S,
        cusparseDnMatDescr_t &A){

    
    double alpha = 1.0, beta = 0.;
    size_t bufsize = 0;

    // Get the buffer size
    assert(cusparseSDDMM_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, A, A, &beta, S,
            CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufsize) == cudaSuccess);

    void *dbuf = NULL;
    assert(cudaMalloc(&dbuf, bufsize) == cudaSuccess);
    assert(cusparseSDDMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, A, A, &beta, S,
            CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, dbuf) == cudaSuccess);

    assert(cudaFree(dbuf) == cudaSuccess);
}

void CusparseAlgo::sddmm(HostSparseMat &S, HostDenseMat &A, HostSparseMat &C){
    // NOTE: S will be modified inplace

    DeviceSparseMat dS;
    DeviceDenseMat dA;

    S.to_device(dS);
    A.to_device(dA);

    cusparseHandle_t handle = NULL;
    assert(cusparseCreate(&handle) == cudaSuccess);

    cusparseSpMatDescr_t S_des;
    cusparseDnMatDescr_t A_des;

    dS.get_cusparse_descriptor(S_des);
    dA.get_cusparse_descriptor(A_des);

    this->sddmm(handle, S_des, A_des);

    // copy the result(modified inplace in dS) back to C
    dS.copy_to_host(C);

    assert(cusparseDestroySpMat(S_des) == cudaSuccess);
    assert(cusparseDestroyDnMat(A_des) == cudaSuccess);
    assert(cusparseDestroy(handle) == cudaSuccess);
}

void CusparseAlgo::sddmm_spmm(
        cusparseHandle_t &handle,
        cusparseSpMatDescr_t &C,
        cusparseDnMatDescr_t &A, 
        cusparseDnMatDescr_t &B,
        cusparseDnMatDescr_t &D, 
        cusparseDnMatDescr_t &E){
    //SDDMM:    (AB) * C    
    double alpha = 1.0, beta = 0.;
    size_t bufsize = 0;
    cusparseConstrainedGeMM_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A, B, &beta, C, CUDA_R_64F,
            &bufsize);
    void *dbuf = NULL;
    assert(cudaMalloc(&dbuf, bufsize) == cudaSuccess);
    assert(cusparseConstrainedGeMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, A, B, &beta, C, CUDA_R_64F, dbuf) == cudaSuccess);
    assert(cudaFree(dbuf) == cudaSuccess);

    //SpMM:     CD = E
    cusparseSpMM_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, C, D, &beta, E, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
            &bufsize);
    // void *dbuf = NULL;
    assert(cudaMalloc(&dbuf, bufsize) == cudaSuccess);
    assert(cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, C, D, &beta, E, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dbuf) == cudaSuccess);
    assert(cudaFree(dbuf) == cudaSuccess);
}

/*********************
Function for Algo
******************/

__global__ void spmm_no_shm_kernel(double *A_vals, int *A_cols, int *A_offsets, int A_nnz, double *B_vals, double *C_vals, int A_h, int A_w, int B_h, int B_w) {
    int gy_C = blockIdx.y * 1 + threadIdx.y, gx_C = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    if(gy_C >= A_h || gx_C >= B_w) return;
    double value = 0.0;
    
    int start_idx = A_offsets[gy_C], end_idx = A_offsets[gy_C+1];

    for(int i = start_idx; i < end_idx; i+=1) {
        int col_A = A_cols[i];
        double val_A = A_vals[i];
        value += val_A * B_vals[col_A*B_w + gx_C];
    }

    C_vals[gy_C*B_w + gx_C] = value;
}

void Algo::spmm_no_shm(HostSparseMat &A, HostDenseMat &B, HostDenseMat &C){
    DeviceSparseMat dA;
    DeviceDenseMat dB, dC;
    A.to_device(dA);
    B.to_device(dB);
    C.to_device(dC);

    int A_h = A.num_rows, A_w = A.num_cols, B_h = B.num_rows, B_w = B.num_cols;
    dim3 dimGrid((B_w+TILE_WIDTH-1)/TILE_WIDTH, A_h);
    dim3 dimBlock(TILE_WIDTH, 1);

    spmm_no_shm_kernel<<<dimGrid, dimBlock>>>(dA.vals, dA.cols, dA.offsets, dA.nnz, dB.vals, dC.vals, A_h, A_w, B_h, B_w);

    dC.copy_to_host(C);
}

__global__ void spmm_kernel(double *A_vals, int *A_cols, int *A_offsets, int A_nnz, double *B_vals, double *C_vals, int A_h, int A_w, int B_h, int B_w) {
    int gy_C = blockIdx.y * 1 + threadIdx.y, gx_C = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    if(gy_C >= A_h || gx_C >= B_w) return;
    double value = 0.0;

    int lx_C = threadIdx.x;
    
    __shared__ int shm_col_A[TILE_WIDTH];
    __shared__ double shm_val_A[TILE_WIDTH];
    int gx_A_start = A_offsets[gy_C], gx_A_end = A_offsets[gy_C+1];
    int n_steps = (gx_A_end-gx_A_start+TILE_WIDTH-1)/(TILE_WIDTH);

    for(int m = 0; m < n_steps; m++) {
        // m is the tile index
        int start_idx = gx_A_start + m * TILE_WIDTH;

        if(start_idx+lx_C < gx_A_end) {
            shm_col_A[lx_C] = A_cols[start_idx+lx_C];
            shm_val_A[lx_C] = A_vals[start_idx+lx_C];
        } else { // out of range => mark value as zero so it will not be counted
            shm_col_A[lx_C] = 0;
            shm_val_A[lx_C] = 0;
        }
        __syncthreads();

        for(int i = 0; i < TILE_WIDTH; i++) {
            value += shm_val_A[i] * B_vals[shm_col_A[i]*B_w + gx_C];
        }
        __syncthreads();
    }

    C_vals[gy_C*B_w + gx_C] = value;
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

    spmm_kernel<<<dimGrid, dimBlock>>>(dA.vals, dA.cols, dA.offsets, dA.nnz, dB.vals, dC.vals, A_h, A_w, B_h, B_w);

    dC.copy_to_host(C);
}


__global__ void sddmm_shm_kernel(double *S_vals, int *S_cols, double *A_vals, double *C_vals, int *tid_to_vid, int *tid_to_rid, int A_w) {
    int lx = threadIdx.x, gx = blockIdx.x * TILE_WIDTH + lx;

    __shared__ double As[TILE_WIDTH];

    int row_C = tid_to_rid[gx];
    double value = 0.0;

    for(int m = 0; m < (A_w + TILE_WIDTH - 1)/(TILE_WIDTH); m++) {
        if(lx+m*TILE_WIDTH < A_w)
            As[lx] = A_vals[row_C*A_w+lx+m*TILE_WIDTH]; // A_vals[row_C][lx+m*TILE_WIDTH]
        else
            As[lx] = 0; // out of border
        
        __syncthreads();
        if(tid_to_vid[gx] != -1) {
            int n_steps = min(TILE_WIDTH, A_w-m*TILE_WIDTH);
            int col_C = S_cols[tid_to_vid[gx]];
            for(int i = 0; i < n_steps; i++) {
                value += As[i] * A_vals[col_C * A_w + i + m * TILE_WIDTH]; // A_vals[row_C][i+m*TILE_WIDTH] * At_vals[i+m*TILE_WIDTH][col_C]
            }
        }
        __syncthreads();
    }

    if(tid_to_vid[gx] != -1)
        C_vals[tid_to_vid[gx]] = S_vals[tid_to_vid[gx]] * value;
}

__global__ void sddmm_kernel(double *S_vals, int *S_cols, int *S_offsets, int S_nnz, double *A_vals, double *C_vals, int A_h, int A_w) {
    int idx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if(idx >= S_nnz) return;
    int col_C = S_cols[idx];
    int row_C;

    int i = 0;
    for(; i < A_h; i++) // find where the index sits
        if(S_offsets[i] > idx)
            break;
    row_C = i-1;


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

    int n_threads = 0;
    for(int i = 0; i < S.num_rows; i++) {
        int start_idx = S.offsets[i], end_idx = S.offsets[i+1];
        n_threads += ((end_idx - start_idx + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    }

    int *tid_to_vid = new int[n_threads]; // thread id to value id
    int *tid_to_rid = new int[n_threads]; // thread id to row id
    int *tid_to_vid_d;
    int *tid_to_rid_d;

    int k = 0;
    for(int i = 0; i < S.num_rows; i++) {
        int start_idx = S.offsets[i], end_idx = S.offsets[i+1];
        for(int j = start_idx; j < end_idx; j++) {
            tid_to_vid[k] = j;
            tid_to_rid[k] = i;
            k += 1;
        }
        for(int j = end_idx; j < start_idx + ((end_idx - start_idx + TILE_WIDTH - 1) / TILE_WIDTH) * (TILE_WIDTH); j++) {
            tid_to_vid[k] = -1;
            tid_to_rid[k] = i;
            k += 1;
        }
    }

    dim3 dimGrid((n_threads+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH);
    cudaMalloc(&tid_to_vid_d, sizeof(int) * n_threads);
    cudaMalloc(&tid_to_rid_d, sizeof(int) * n_threads);
    cudaMemcpy(tid_to_vid_d, tid_to_vid, sizeof(int) * n_threads, cudaMemcpyHostToDevice);
    cudaMemcpy(tid_to_rid_d, tid_to_rid, sizeof(int) * n_threads, cudaMemcpyHostToDevice);

    sddmm_shm_kernel<<<dimGrid, dimBlock>>>(dS.vals, dS.cols, dA.vals, dC.vals, tid_to_vid_d, tid_to_rid_d, dA.num_cols);

    free(tid_to_vid);
    free(tid_to_rid);
    cudaFree(tid_to_vid_d);
    cudaFree(tid_to_rid_d);
    dC.copy_to_host(C);

}

void Algo::sddmm_block_over_nnz(HostSparseMat &S, HostDenseMat &A, HostSparseMat &C){
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

__global__ void count_num_blocks_in_each_row(
        int S_num_rows, int *S_offsets, int *block_offsets){

    int row_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(row_id >= S_num_rows) return;

    int nnz_col = S_offsets[row_id+1] - S_offsets[row_id];
    // += the number of blocks needed for this row
    atomicAdd(&block_offsets[row_id+1], (nnz_col + TILE_WIDTH - 1) / TILE_WIDTH);
}

__global__ void sddmm_block_over_nnz_in_same_row_kernel(
    int S_num_rows, int *S_offsets, int *S_cols, double *S_vals,
    int A_num_cols, double *A_vals,
    double *C_vals,
    int *block_offsets){

    // Each block must first search which row is belongs to by binary search
    int block_idx = blockIdx.x;
    int l = 0, r = S_num_rows;
    int mid;

    __shared__ int shm_row_idx;
    // only one thread needs to compute this
    if(threadIdx.x == 0){
        // l = upperbound(block_offsets, block_idx)
        // find the smallest(first) idx s.t. block_offsets[idx] > block_idx
        while(l < r){
            mid = (l + r) / 2;
            if(block_offsets[mid] <= block_idx){
                l = mid+1;
            }else{
                r = mid;
            }
        }
        assert(l <= S_num_rows);
        shm_row_idx = l-1;
    }
    __syncthreads();

    int row_idx = shm_row_idx; // copy to this thread's private space
    int start = S_offsets[row_idx], end = S_offsets[row_idx+1];

    int _j = (blockIdx.x - block_offsets[row_idx]) * TILE_WIDTH + threadIdx.x;
    int j = -1;
    if(start + _j < end) j = S_cols[start + _j];

    __shared__ double A_shm[TILE_WIDTH];

    double value = 0.;
    for(int k = 0; k < A_num_cols; k += TILE_WIDTH){
        int my_k = k + threadIdx.x;
        if(my_k < A_num_cols){
            A_shm[threadIdx.x] = A_vals[row_idx * A_num_cols + my_k];
        }
        __syncthreads();

        const int bound_tile_width = MIN(TILE_WIDTH, A_num_cols - k);

        if(j != -1){
            for(int kk = 0; kk < bound_tile_width; ++kk){
                value += A_shm[kk] * A_vals[j * A_num_cols + k + kk];
            }
        }
        __syncthreads();
    }

    // Write back
    if(j != -1) C_vals[start + _j] = S_vals[start + _j] * value;
}
void Algo::sddmm_block_over_nnz_but_in_same_row(HostSparseMat &S, HostDenseMat &A, HostSparseMat &C){
    DeviceSparseMat dS, dC;
    DeviceDenseMat dA;

    S.to_device(dS);
    A.to_device(dA);
    C.to_device(dC);

    // block_offsets[row_id] = # of blocks needed this row
    int *block_offsets;

    assert(cudaMalloc(&block_offsets, sizeof(int) * (S.num_rows+1)) == cudaSuccess);
    // set 0 initially
    assert(cudaMemset(block_offsets, 0, sizeof(int) * (S.num_rows+1)) == cudaSuccess);

    const int num_threads = 256;
    // Parallelize over # of rows
    count_num_blocks_in_each_row<<<(S.num_rows + num_threads - 1)/ num_threads, num_threads>>>(
                S.num_rows, dS.offsets, block_offsets);
    // prefix sum
    // block_offsets[i] = # of blocks that are in [0, i) rows
    thrust::device_ptr<int> ptr(block_offsets);
    thrust::inclusive_scan(ptr+1, ptr+S.num_rows+1, ptr+1);

    int num_blocks = 0;

    // Only copy the total number of blocks back
    cudaMemcpy(&num_blocks, block_offsets + S.num_rows, sizeof(int), cudaMemcpyDeviceToHost);

    sddmm_block_over_nnz_in_same_row_kernel<<<num_blocks, TILE_WIDTH>>>(
            S.num_rows, dS.offsets, dS.cols, dS.vals,
            A.num_cols, dA.vals,
            dC.vals, block_offsets);

    dC.copy_to_host(C);

    assert(cudaFree(block_offsets) == cudaSuccess);
}


__global__ void sddmm_launch_kernel_as_dense_matrix_kernel(
        int S_num_rows, int *S_offsets, int *S_cols, double *S_vals,
        int A_num_cols, double *A_vals,
        double *C_vals){

    int i = blockIdx.x;
    int _j_first = blockIdx.y * blockDim.y;
    int _j = blockIdx.y * blockDim.y + threadIdx.y;

    int start = S_offsets[i], end = S_offsets[i+1];
    // if the first thread in this block has nothing to do,
    // this block has no work to do
    if(start + _j_first >= end) return;

    int j = (start + _j < end)? (S_cols[start + _j]):(-1);

    __shared__ double A_shm[TILE_WIDTH];

    double value = 0.;
    for(int k = 0; k < A_num_cols; k += TILE_WIDTH){
        int my_k = k + threadIdx.y;
        if(my_k < A_num_cols){
            A_shm[threadIdx.y] = A_vals[i * A_num_cols + my_k];
        }
        __syncthreads();

        const int bound_tile_width = MIN(TILE_WIDTH, A_num_cols - k);

        if(j != -1){
            for(int kk = 0; kk < bound_tile_width; ++kk){
                value += A_shm[kk] * A_vals[j * A_num_cols + (k + kk)];
            }
        }
        __syncthreads();
    }

    // Write to C
    if(j != -1) C_vals[start + _j] = S_vals[start + _j] * value;
}

void Algo::sddmm_launch_kernel_as_dense_matrix(
        HostSparseMat &S, HostDenseMat &A, HostSparseMat &C){

    DeviceSparseMat dS, dC;
    DeviceDenseMat dA;

    S.to_device(dS);
    A.to_device(dA);
    C.to_device(dC);

    // Launch the kernel as if it is a dense matrix
    dim3 threadsPerBlock(1, TILE_WIDTH);
    dim3 numBlocks(S.num_rows, (S.num_cols + TILE_WIDTH - 1) / TILE_WIDTH);

    sddmm_launch_kernel_as_dense_matrix_kernel<<<numBlocks, threadsPerBlock>>>(
        S.num_rows, dS.offsets, dS.cols, dS.vals,
        A.num_cols, dA.vals,
        dC.vals);

    dC.copy_to_host(C);
}

__global__ void sddmm_block_over_nnz_if_same_row_use_shm_kernel(
        int S_num_rows, int S_nnz, int *S_offsets, int *S_cols, double *S_vals,
        int A_num_cols, double *A_vals,
        double *C_vals){

    int _j = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int bound = MIN(TILE_WIDTH, S_nnz - blockIdx.x * TILE_WIDTH);

    __shared__ int row_indices[TILE_WIDTH];
    __shared__ double A_shm[TILE_WIDTH];

    // find this element's row idx (i.e. i)
    int l = 0, r = S_num_rows;
    int mid;

    // Binary search to find the row idx
    if(threadIdx.x < bound){
        while(l < r){
            mid = (l+r)/2;
            if(S_offsets[mid] <= _j){
                l = mid+1;
            }else{
                r = mid;
            }
        }
        assert(l <= S_num_rows);
        row_indices[threadIdx.x] = l-1;
    }
    __syncthreads();

    double value = 0.;
    // If the first row_idx and the last row_idx are the same,
    // we can use shared memory (and all threads will enter this branch)
    if(row_indices[0] == row_indices[bound-1]){
        int i = row_indices[0]; // load from the shared mem
        int j = (threadIdx.x < bound)?(S_cols[_j]):(-1);

        for(int k = 0; k < A_num_cols; k += TILE_WIDTH){
            int my_k = k + threadIdx.x;
            if(my_k < A_num_cols){
                A_shm[threadIdx.x] = A_vals[i * A_num_cols + my_k];
            }
            __syncthreads();
            if(j != -1){
                const int bound_tile_width = MIN(TILE_WIDTH, A_num_cols - k);
                for(int kk = 0; kk < bound_tile_width; ++kk){
                    value += A_shm[kk] * A_vals[j * A_num_cols + (k + kk)];
                }
            }
            __syncthreads();
        }
    }else if(threadIdx.x < bound){
        // Otherwise, we cannot use shared memory to accelerate
        // in this case, each thread will compute by its own element without collaborating
        int i = l-1;
        int j = S_cols[_j];
        for(int k = 0; k < A_num_cols; ++k){
            value += A_vals[i * A_num_cols + k] * A_vals[j * A_num_cols + k];
        }
    }
    // Write to C
    if(threadIdx.x < bound) C_vals[_j] = S_vals[_j] * value;
}

void Algo::sddmm_block_over_nnz_if_same_row_use_shm(
        HostSparseMat &S, HostDenseMat &A, HostSparseMat &C){

    DeviceSparseMat dS, dC;
    DeviceDenseMat dA;

    S.to_device(dS);
    A.to_device(dA);
    C.to_device(dC);

    dim3 threadsPerBlock(TILE_WIDTH);
    dim3 numBlocks((S.nnz + TILE_WIDTH - 1) / TILE_WIDTH);

    sddmm_block_over_nnz_if_same_row_use_shm_kernel<<<numBlocks, threadsPerBlock>>>(
        S.num_rows, S.nnz, dS.offsets, dS.cols, dS.vals,
        A.num_cols, dA.vals,
        dC.vals);

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
