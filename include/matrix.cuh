#pragma once
#include <stdio.h>
#include <cusparse.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#ifndef ZERO_RATIO
#define ZERO_RATIO 0.7
#endif

struct HostDenseMat;
struct DeviceDenseMat;

struct HostSparseMat;
struct DeviceSparseMat;

struct HostDenseMat{
    int num_rows, num_cols;
    double *vals;
    bool to_delete;
    HostDenseMat(int num_rows_, int num_cols_, double *vals_, bool to_delete_);
    ~HostDenseMat();
    void to_device(DeviceDenseMat &d);
    void to_sparse(HostSparseMat &m);
    friend std::ostream& operator<<(std::ostream &os, const HostDenseMat &obj);
    // overload set
    double & operator [](int i) {return vals[i];}
    // overload get
    double operator [](int i) const {return vals[i];};
    // overload comparison
    bool operator == (const HostDenseMat& m2) {
        if(num_rows != m2.num_rows) return false;
        if(num_cols != m2.num_cols) return false;
        double epsilon = 1e-4; // std::numeric_limits<double>::epsilon();
        for(int i = 0; i < num_rows; i++)
            for(int j = 0; j < num_cols; j++)
                if(std::fabs(vals[i*num_cols+j] - m2.vals[i*num_cols+j]) > epsilon)
                    return false;
        return true;
    }
    bool operator != (const HostDenseMat& m2) {
        return !(*this == m2);
    }
};

struct DeviceDenseMat{
    int num_rows, num_cols;
    double *vals;
    DeviceDenseMat() = default;
    ~DeviceDenseMat();

    void get_cusparse_descriptor(cusparseDnMatDescr_t &mat);
    void copy_to_host(HostDenseMat &h);
    // overload set
    __device__ double & operator [](int i) {return vals[i];}
    // overload get
    __device__ double operator [](int i) const {return vals[i];};
};

struct HostSparseMat{
    int num_rows, num_cols;
    int nnz;
    int *offsets;
    int *cols;
    double *vals;
    bool to_delete;
    friend std::ostream& operator<<(std::ostream &os, const HostSparseMat &obj);
    HostSparseMat(
            int num_rows_, int num_cols_, int nnz_,
            int *offsets_, int *cols_, double *vals_, bool to_delete_);
    ~HostSparseMat();
    void to_device(DeviceSparseMat &d);
    void to_dense(HostDenseMat &mat);
    bool operator==(const HostSparseMat &r);
};

struct DeviceSparseMat{
    int num_rows, num_cols;
    int nnz;
    int *offsets;
    int *cols;
    double *vals;

    DeviceSparseMat() = default;
    DeviceSparseMat(int num_rows_, int num_cols_, int nnz_,
            int *offsets_, int *cols_, double *vals_);
    ~DeviceSparseMat();
    void get_cusparse_descriptor(cusparseSpMatDescr_t &mat);
    void copy_to_host(HostSparseMat &h);
};

struct MatrixGenerator{
    void generate_sparse_csr(int, int, int&, int**, int**, double**);
    void generate_binary_sparse_csr(int, int, int&, int**, int**, double**);
    void generate_dense(int, int, double**);
};
