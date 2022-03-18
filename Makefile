.PHONY = all

CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include

all:
	nvcc $(INC) cusparse_benchmark.cu -lcusparse -lcudart -o cusparse_benchmark
