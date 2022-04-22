# reference: https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable

.PHONY = all

CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include

SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include
OBJS = $(OBJ_DIR)/algo.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/benchmark.o $(OBJ_DIR)/checker.o
EXEC = main

all: $(OBJS) main.o
	nvcc -arch sm_60 $(INC) main.o $(OBJS) -lcusparse -lcudart -lcudadevrt -o $(EXEC)
main.o: main.cu
	nvcc -arch sm_60 -rdc true -c $< -o $@
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	nvcc -arch sm_60 -rdc true -c $< -o $@
clean:
	rm bin/* *.o $(EXEC)
