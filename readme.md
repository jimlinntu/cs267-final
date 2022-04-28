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