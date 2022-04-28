#include "../include/benchmark.cuh"

double avg(std::vector<double> &v){
    double a = 0;
    for(double &e: v) a += e;
    return a / (double)v.size();
}

void Benchmarker::benchmark_sddmm(BenchmarkResult &bresult){
    const int S_num_rows = 8023;
    const int A_num_cols = 1049;

    Algo algo;
    CusparseAlgo cualgo;

    MatrixGenerator mg;

    std::map<std::string, std::vector<double>> m;
    std::map<std::string, std::vector<double>> gpu_m;
    clock_t start, end;
    float gpu_time;

    for(int i = 0; i < NUMEXPS; ++i){
        int S_nnz;
        int *S_offsets;
        int *S_cols;
        double *S_vals;
        mg.generate_binary_sparse_csr(S_num_rows, S_num_rows, S_nnz, &S_offsets, &S_cols, &S_vals);

        double *A_vals;
        mg.generate_dense(S_num_rows, A_num_cols, &A_vals);

        int *C_offsets = new int[S_num_rows+1];
        int *C_cols = new int[S_nnz];
        double *C_vals = new double[S_nnz];
        double *C_vals_cusparse = new double[S_nnz];

        memcpy(C_offsets, S_offsets, (S_num_rows+1) * sizeof(int));
        memcpy(C_cols, S_cols, (S_nnz) * sizeof(int));

        HostSparseMat S(S_num_rows, S_num_rows, S_nnz, S_offsets, S_cols, S_vals, true);
        HostDenseMat A(S_num_rows, A_num_cols, A_vals, true);
        HostSparseMat C(S_num_rows, S_num_rows, S_nnz, C_offsets, C_cols, C_vals, true);

        start = clock();
        algo.sddmm_with_tid_mapping(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_with_tid_mapping"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_with_tid_mapping"].push_back(gpu_time);

        start = clock();
        algo.sddmm_by_dgemm(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_by_dgemm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_by_dgemm"].push_back(gpu_time);

        start = clock();
        algo.sddmm_block_over_nnz(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_block_over_nnz_wo_shm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_block_over_nnz_wo_shm"].push_back(gpu_time);

        start = clock();
        algo.sddmm_block_over_nnz_but_in_same_row(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_block_over_nnz_but_in_same_row"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_block_over_nnz_but_in_same_row"].push_back(gpu_time);

        start = clock();
        algo.sddmm_launch_kernel_as_dense_matrix(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_launch_kernel_as_dense_matrix"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_launch_kernel_as_dense_matrix"].push_back(gpu_time);

        start = clock();
        algo.sddmm_block_over_nnz_if_same_row_use_shm(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_block_over_nnz_if_same_row_use_shm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_block_over_nnz_if_same_row_use_shm"].push_back(gpu_time);

        start = clock();
        algo.sddmm_dynamic_parallelism(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_dynamic_parallelism"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_dynamic_parallelism"].push_back(gpu_time);

        start = clock();
        cualgo.sddmm(S, A, C, &gpu_time);
        end = clock();
        m["cusparsesddmm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["cusparsesddmm"].push_back(gpu_time);
    }

    bresult.result["sddmm_with_tid_mapping"] = avg(m["sddmm_with_tid_mapping"]);
    bresult.result["sddmm_by_dgemm"] = avg(m["sddmm_by_dgemm"]);
    bresult.result["sddmm_block_over_nnz_wo_shm"] = avg(m["sddmm_block_over_nnz_wo_shm"]);
    bresult.result["sddmm_block_over_nnz_but_in_same_row"] = avg(m["sddmm_block_over_nnz_but_in_same_row"]);
    bresult.result["sddmm_launch_kernel_as_dense_matrix"] = avg(m["sddmm_launch_kernel_as_dense_matrix"]);
    bresult.result["sddmm_block_over_nnz_if_same_row_use_shm"] = avg(m["sddmm_block_over_nnz_if_same_row_use_shm"]);
    bresult.result["sddmm_dynamic_parallelism"] = avg(m["sddmm_dynamic_parallelism"]);
    bresult.result["cusparsesddmm"] = avg(m["cusparsesddmm"]);

    bresult.gpu_compute_result["sddmm_with_tid_mapping"] = avg(gpu_m["sddmm_with_tid_mapping"]);
    bresult.gpu_compute_result["sddmm_by_dgemm"] = avg(gpu_m["sddmm_by_dgemm"]);
    bresult.gpu_compute_result["sddmm_block_over_nnz_wo_shm"] = avg(gpu_m["sddmm_block_over_nnz_wo_shm"]);
    bresult.gpu_compute_result["sddmm_block_over_nnz_but_in_same_row"] = avg(gpu_m["sddmm_block_over_nnz_but_in_same_row"]);
    bresult.gpu_compute_result["sddmm_launch_kernel_as_dense_matrix"] = avg(gpu_m["sddmm_launch_kernel_as_dense_matrix"]);
    bresult.gpu_compute_result["sddmm_block_over_nnz_if_same_row_use_shm"] = avg(gpu_m["sddmm_block_over_nnz_if_same_row_use_shm"]);
    bresult.gpu_compute_result["sddmm_dynamic_parallelism"] = avg(gpu_m["sddmm_dynamic_parallelism"]);
    bresult.gpu_compute_result["cusparsesddmm"] = avg(gpu_m["cusparsesddmm"]);
}

void Benchmarker::benchmark_spmm(BenchmarkResult &bresult){
    const int S_num_rows = 8023;
    const int S_num_cols = 3928;
    const int A_num_rows = S_num_cols;
    const int A_num_cols = 1049;

    Algo algo;
    CusparseAlgo cualgo;

    MatrixGenerator mg;

    std::map<std::string, std::vector<double>> m;
    std::map<std::string, std::vector<double>> gpu_m;
    clock_t start, end;
    float gpu_time;

    for(int i = 0; i < NUMEXPS; ++i){
        int S_nnz;
        int *S_offsets;
        int *S_cols;
        double *S_vals;
        mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);

        double *A_vals;
        mg.generate_dense(A_num_rows, A_num_cols, &A_vals);

        // Create the output dense matrix
        double *C_vals = new double[S_num_rows * A_num_cols];

        HostSparseMat S(S_num_rows, S_num_cols, S_nnz, S_offsets, S_cols, S_vals, true);
        HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);
        HostDenseMat C(S_num_rows, A_num_cols, C_vals, true);

        start = clock();
        algo.spmm_with_shm(S, A, C, &gpu_time);
        end = clock();
        m["spmm_with_shm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["spmm_with_shm"].push_back(gpu_time);

        start = clock();
        algo.spmm_no_shm(S, A, C, &gpu_time);
        end = clock();
        m["spmm_no_shm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["spmm_no_shm"].push_back(gpu_time);

        start = clock();
        algo.spmm_with_shm_jim(S, A, C, &gpu_time);
        end = clock();
        m["spmm_with_shm_jim"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["spmm_with_shm_jim"].push_back(gpu_time);
        
        start = clock();
        algo.spmm_by_dgemm(S, A, C, &gpu_time);
        end = clock();
        m["spmm_by_dgemm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["spmm_by_dgemm"].push_back(gpu_time);

        start = clock();
        algo.spmm_with_shm_jim_transpose_first(S, A, C, &gpu_time);
        end = clock();
        m["spmm_with_shm_jim_transpose_first"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["spmm_with_shm_jim_transpose_first"].push_back(gpu_time);

        start = clock();
        cualgo.spmm(S, A, C, &gpu_time);
        end = clock();
        m["cusparsespmm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["cusparsespmm"].push_back(gpu_time);
    }

    bresult.result["spmm_with_shm"] = avg(m["spmm_with_shm"]);
    bresult.result["spmm_no_shm"] = avg(m["spmm_no_shm"]);
    bresult.result["spmm_with_shm_jim"] = avg(m["spmm_with_shm_jim"]);
    bresult.result["spmm_with_shm_jim_transpose_first"] = avg(m["spmm_with_shm_jim_transpose_first"]);
    bresult.result["spmm_by_dgemm"] = avg(m["spmm_by_dgemm"]);
    bresult.result["cusparsespmm"] = avg(m["cusparsespmm"]);

    bresult.gpu_compute_result["spmm_with_shm"] = avg(gpu_m["spmm_with_shm"]);
    bresult.gpu_compute_result["spmm_no_shm"] = avg(gpu_m["spmm_no_shm"]);
    bresult.gpu_compute_result["spmm_with_shm_jim"] = avg(gpu_m["spmm_with_shm_jim"]);
    bresult.gpu_compute_result["spmm_with_shm_jim_transpose_first"] = avg(gpu_m["spmm_with_shm_jim_transpose_first"]);
    bresult.gpu_compute_result["spmm_by_dgemm"] = avg(gpu_m["spmm_by_dgemm"]);
    bresult.gpu_compute_result["cusparsespmm"] = avg(gpu_m["cusparsespmm"]);
}

void Benchmarker::benchmark_sddmm_spmm(BenchmarkResult &bresult){
    const int S_num_rows = 8023;
    const int S_num_cols = S_num_rows;
    const int A_num_rows = S_num_rows;
    const int A_num_cols = 1049;

    Algo algo;
    CusparseAlgo cualgo;

    MatrixGenerator mg;

    std::map<std::string, std::vector<double>> m;
    std::map<std::string, std::vector<double>> gpu_m;
    clock_t start, end;
    float gpu_time;

    for(int i = 0; i < NUMEXPS; ++i){
        int S_nnz;
        int *S_offsets;
        int *S_cols;
        double *S_vals;
        mg.generate_sparse_csr(S_num_rows, S_num_cols, S_nnz, &S_offsets, &S_cols, &S_vals);

        double *A_vals;
        mg.generate_dense(A_num_rows, A_num_cols, &A_vals);

        // Create the output dense matrix
        double *C_vals = new double[S_num_rows * A_num_cols];

        HostSparseMat S(S_num_rows, S_num_cols, S_nnz, S_offsets, S_cols, S_vals, true);
        HostDenseMat A(A_num_rows, A_num_cols, A_vals, true);
        HostDenseMat C(S_num_rows, A_num_cols, C_vals, true);

        start = clock();
        algo.sddmm_spmm_block_over_sparse_launch_as_dense_matrix(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_spmm_block_over_sparse_launch_as_dense_matrix"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_spmm_block_over_sparse_launch_as_dense_matrix"].push_back(gpu_time);

        // This one is extremely slow!! (because of duplicate works)
        if(0){
            start = clock();
            algo.sddmm_spmm_block_over_output(S, A, C);
            end = clock();
            m["sddmm_spmm_block_over_output"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        }

        start = clock();
        algo.sddmm_spmm_naive_back2back_calls(S, A, C, &gpu_time);
        end = clock();
        m["sddmm_spmm_naive_back2back_calls"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["sddmm_spmm_naive_back2back_calls"].push_back(gpu_time);

        start = clock();
        cualgo.sddmm_spmm(S, A, C, &gpu_time);
        end = clock();
        m["cusparse_sddmm_spmm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
        gpu_m["cusparse_sddmm_spmm"].push_back(gpu_time);
    }

    bresult.result["sddmm_spmm_block_over_sparse_launch_as_dense_matrix"] = avg(m["sddmm_spmm_block_over_sparse_launch_as_dense_matrix"]);
    /* bresult.result["sddmm_spmm_block_over_output"] = avg(m["sddmm_spmm_block_over_output"]); */
    bresult.result["sddmm_spmm_naive_back2back_calls"] = avg(m["sddmm_spmm_naive_back2back_calls"]);
    bresult.result["cusparse_sddmm_spmm"] = avg(m["cusparse_sddmm_spmm"]);

    bresult.gpu_compute_result["sddmm_spmm_block_over_sparse_launch_as_dense_matrix"] =\
        avg(gpu_m["sddmm_spmm_block_over_sparse_launch_as_dense_matrix"]);
    bresult.gpu_compute_result["sddmm_spmm_naive_back2back_calls"] = avg(gpu_m["sddmm_spmm_naive_back2back_calls"]);
    bresult.gpu_compute_result["cusparse_sddmm_spmm"] =\
        avg(gpu_m["cusparse_sddmm_spmm"]);
}

std::ostream& operator<<(std::ostream &os, const BenchmarkResult &obj){
    os << "[*] CPU -> GPU -> CPU time:\n";
    auto &result = obj.result;
    for(auto it = result.begin(); it != result.end(); ++it){
        std::string expname = it->first;
        double avg_sec = it->second;

        os << expname << " takes " << avg_sec << " seconds\n";
    }

    os << "[*] Pure GPU compute time:\n";
    auto &gpu_compute_result = obj.gpu_compute_result;
    for(auto it = gpu_compute_result.begin(); it != gpu_compute_result.end(); ++it){
        std::string expname = it->first;
        double avg_sec = it->second;

        os << expname << " takes " << avg_sec << " seconds\n";
    }
    return os;
}
