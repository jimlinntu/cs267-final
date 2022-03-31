.PHONY = all

CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include

all:
	nvcc $(INC) main.cu -lcusparse -lcudart -o main
