# Accelerate SDDMM\_SpMM in CUDA C/C++

## Environment Setup on Cori

```
. ./set_cori_env.sh
salloc -C gpu -t 1:00:00 -c 10 -G 1 -q interactive -A mp309
```

## How to Run
```
make
./main
```