#include "../include/checker.cuh"

void Checker::check_correctness_sddmm() {
    const int num_testcases = 100;

    MatrixGenerator mg;
    // Initialize algorithms
    Algo algo;
    CusparseAlgo cualgo;

    for(int i = 0; i < num_testcases; ++i){
        // [100, 1000)
        int l = 100, r = 1000;
        int S_num_rows = l + rand() % (r-l);
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

        if(false){
            std::fill(C.vals, C.vals+S_nnz, 0);
            algo.sddmm_seq(S, A, C); // this one is pretty slow, turn it true if you want to test it
            assert(C == C_cusparse);
        }

        std::fill(C.vals, C.vals+S_nnz, 0);
        algo.sddmm_block_over_nnz(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_nnz, 0);
        algo.sddmm_with_tid_mapping(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_nnz, 0);
        algo.sddmm_launch_kernel_as_dense_matrix(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_nnz, 0);
        algo.sddmm_block_over_nnz_if_same_row_use_shm(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_nnz, 0);
        algo.sddmm_dynamic_parallelism(S, A, C);
        assert(C == C_cusparse);

        // Clean up
        delete[] C_vals_cusparse;
    }
}

void Checker::check_correctness_spmm() {

    const int num_testcases = 100;

    MatrixGenerator mg;
    // Initialize algorithms
    Algo algo;
    CusparseAlgo cualgo;

    for(int i = 0; i < num_testcases; ++i){
        // [100, 1000)
        int l = 100, r = 1000;
        int S_num_rows = l + rand() % (r-l);
        int S_num_cols = l + rand() % (r-l);
        int A_num_rows = S_num_cols;
        int A_num_cols = l + rand() % (r-l);

        // Generate a sparse matrix
        int S_nnz;
        int *S_offsets;
        int *S_cols;
        double *S_vals;

        mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);

        double *A_vals;
        // Generate a dense matrix
        mg.generate_dense(A_num_rows, A_num_cols, &A_vals);

        // Create the output dense matrix
        double *C_vals = new double[S_num_rows * A_num_cols];
        double *C_vals_cusparse = new double[S_num_rows * A_num_cols];

        HostSparseMat S(S_num_rows, S_num_cols, S_nnz, S_offsets, S_cols, S_vals, true);
        HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);
        HostDenseMat C(S_num_rows, A_num_cols, C_vals, true);
        HostDenseMat C_cusparse(S_num_rows, A_num_cols, C_vals_cusparse, true);

        // Run cusparsesddmm as the ground truth
        cualgo.spmm(S, A, C_cusparse);

        // Test a bunch of functions here:
        std::fill(C.vals, C.vals+S_num_rows*A_num_cols, 0);
        algo.spmm_with_shm(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_num_rows*A_num_cols, 0);
        algo.spmm_no_shm(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_num_rows*A_num_cols, 0);
        algo.spmm_with_shm_jim(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_num_rows*A_num_cols, 0);
        algo.spmm_with_shm_jim_transpose_first(S, A, C);
        assert(C == C_cusparse);
    }
}

void Checker::check_correctness_sddmm_spmm(){
    const int num_testcases = 100;

    MatrixGenerator mg;
    // Initialize algorithms
    Algo algo;
    CusparseAlgo cualgo;

    for(int i = 0; i < num_testcases; ++i){
        // [100, 1000)
        int l = 100, r = 1000;
        int S_num_rows = l + rand() % (r-l);
        int A_num_cols = l + rand() % (r-l);

        // Generate a sparse matrix
        int S_nnz;
        int *S_offsets;
        int *S_cols;
        double *S_vals;

        mg.generate_binary_sparse_csr(S_num_rows, S_num_rows, S_nnz, &S_offsets, &S_cols, &S_vals);

        double *A_vals;
        // Generate a dense matrix
        mg.generate_dense(S_num_rows, A_num_cols, &A_vals);

        // Create the output dense matrix
        double *C_vals = new double[S_num_rows * A_num_cols];
        double *C_vals_cusparse = new double[S_num_rows * A_num_cols];

        HostSparseMat S(S_num_rows, S_num_rows, S_nnz, S_offsets, S_cols, S_vals, true);

        HostDenseMat A(S_num_rows, A_num_cols, A_vals, true);
        HostDenseMat C(S_num_rows, A_num_cols, C_vals, true);

        HostDenseMat C_cusparse(S_num_rows, A_num_cols, C_vals_cusparse, true);

        cualgo.sddmm_spmm(S, A, C_cusparse);

        algo.sddmm_spmm_block_over_sparse_launch_as_dense_matrix(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_num_rows*A_num_cols, 0);
        algo.sddmm_spmm_block_over_output(S, A, C);
        assert(C == C_cusparse);

        std::fill(C.vals, C.vals+S_num_rows*A_num_cols, 0);
        algo.sddmm_spmm_naive_back2back_calls(S, A, C);
        assert(C == C_cusparse);
    }
}
