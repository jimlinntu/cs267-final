#include "../include/benchmark.cuh"

double avg(std::vector<double> &v){
    double a = 0;
    for(double &e: v) a += e;
    return a / (double)v.size();
}

void Benchmarker::benchmark_sddmm(BenchmarkResult &bresult){
    const int S_num_rows = 4093;
    const int A_num_cols = 1049;

    Algo algo;
    CusparseAlgo cualgo;

    MatrixGenerator mg;

    std::map<std::string, std::vector<double>> m;
    clock_t start, end;

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
        algo.sddmm_block_over_nnz(S, A, C);
        end = clock();
        m["sddmm_block_over_nnz_wo_shm"].push_back((double)(end - start) / CLOCKS_PER_SEC);

        start = clock();
        algo.sddmm_block_over_nnz_but_in_same_row(S, A, C);
        end = clock();
        m["sddmm_block_over_nnz_but_in_same_row"].push_back((double)(end - start) / CLOCKS_PER_SEC);

        start = clock();
        cualgo.sddmm(S, A, C);
        end = clock();
        m["cusparsesddmm"].push_back((double)(end - start) / CLOCKS_PER_SEC);
    }

    bresult.result["sddmm_block_over_nnz_wo_shm"] = avg(m["sddmm_block_over_nnz_wo_shm"]);
    bresult.result["sddmm_block_over_nnz_but_in_same_row"] = avg(m["sddmm_block_over_nnz_but_in_same_row"]);
    bresult.result["cusparsesddmm"] = avg(m["cusparsesddmm"]);
}

std::ostream& operator<<(std::ostream &os, const BenchmarkResult &obj){
    auto &result = obj.result;
    for(auto it = result.begin(); it != result.end(); ++it){
        std::string expname = it->first;
        double avg_sec = it->second;

        os << expname << " takes " << avg_sec << " seconds\n";
    }
}
