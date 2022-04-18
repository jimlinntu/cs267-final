#include "../include/algo.cuh"
#include "../include/matrix.cuh"

#include <string>
#include <map>
#include <iostream>
#include <vector>

#define NUMEXPS 10

struct BenchmarkResult;
struct Benchmarker;

struct BenchmarkResult{
    using ExpName = std::string;
    using Sec = double;
    std::map<ExpName, Sec> result;
    void to_csv(); // TODO: save the experiment result
    friend std::ostream& operator<<(std::ostream &os, const BenchmarkResult &obj);
};

struct Benchmarker{
    void benchmark_sddmm(BenchmarkResult &bresult);
};
