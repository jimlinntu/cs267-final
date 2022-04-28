# Accelerate SDDMM\_SpMM in CUDA C/C++

## Environment Setup on Cori

```bash
. ./set_cori_env.sh
salloc -C gpu -t 1:00:00 -c 10 -G 1 -q interactive -A mp309
```

## How to Run
```bash
make
./main
```

## How to Run with Customized Macro Variable
```bash
make clean # important: must clean *.o file before we make again
make CPPFLAGS="-DZERO_RATIO=0.6 -DTILE_WIDTH=32"
```

## How to Run with Customized Dimension
```bash
./main <S_height> <A_width>
```

## Output Example
```bash
===== Matrix Dimension =====
S_height: 4096
S_width: 4096
A_height: 4096
A_width: 3152
===== Macro Variable Setting =====
ZERO_RATIO: 0.7
TILE_WIDTH: 16
===== PASS sddmm correctness check =====
===== PASS spmm correctness check =====
===== PASS sddmm_spmm correctness check =====
====== sddmm benchmark result: =====
[*] CPU -> GPU -> CPU time:
cusparsesddmm takes 0.357644 seconds
sddmm_block_over_nnz_but_in_same_row takes 0.617325 seconds
sddmm_block_over_nnz_if_same_row_use_shm takes 0.620393 seconds
sddmm_block_over_nnz_wo_shm takes 0.635245 seconds
sddmm_by_dgemm takes 0.711392 seconds
sddmm_dynamic_parallelism takes 1.12271 seconds
sddmm_launch_kernel_as_dense_matrix takes 0.473047 seconds
sddmm_with_tid_mapping takes 0.573721 seconds
[*] Pure GPU compute time:
cusparsesddmm takes 0.33836 seconds
sddmm_block_over_nnz_but_in_same_row takes 0.591711 seconds
sddmm_block_over_nnz_if_same_row_use_shm takes 0.594982 seconds
sddmm_block_over_nnz_wo_shm takes 0.609888 seconds
sddmm_by_dgemm takes 0.343539 seconds
sddmm_dynamic_parallelism takes 1.08905 seconds
sddmm_launch_kernel_as_dense_matrix takes 0.447455 seconds
sddmm_with_tid_mapping takes 0.520702 seconds
====================================
====== spmm benchmark result: =====
[*] CPU -> GPU -> CPU time:
cusparsespmm takes 0.158922 seconds
spmm_by_dgemm takes 0.460506 seconds
spmm_no_shm takes 0.332932 seconds
spmm_with_shm takes 0.357352 seconds
spmm_with_shm_jim takes 0.247449 seconds
spmm_with_shm_jim_transpose_first takes 0.509856 seconds
[*] Pure GPU compute time:
cusparsespmm takes 0.123755 seconds
spmm_by_dgemm takes 0.344797 seconds
spmm_no_shm takes 0.298475 seconds
spmm_with_shm takes 0.301093 seconds
spmm_with_shm_jim takes 0.213018 seconds
spmm_with_shm_jim_transpose_first takes 0.475655 seconds
====================================
=== sddmm_spmm benchmark result: ===
[*] CPU -> GPU -> CPU time:
cusparse_sddmm_spmm takes 0.499397 seconds
sddmm_spmm_block_over_sparse_launch_as_dense_matrix takes 0.586286 seconds
sddmm_spmm_by_dgemm takes 1.20221 seconds
sddmm_spmm_naive_back2back_calls takes 0.697983 seconds
[*] Pure GPU compute time:
cusparse_sddmm_spmm takes 0.464935 seconds
sddmm_spmm_block_over_sparse_launch_as_dense_matrix takes 0.552363 seconds
sddmm_spmm_by_dgemm takes 0.69091 seconds
sddmm_spmm_naive_back2back_calls takes 0.6634 seconds
====================================
```