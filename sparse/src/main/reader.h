#ifndef SRC_MAIN_READER_H_
#define SRC_MAIN_READER_H_

using custom_int = long long int; // when hypre is installed with --enable-bigint
//using custom_int = int;         // when hypre is installed with default configuration

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

struct mtrx_csr {
    std::vector<double> values_;          // common
    std::vector<custom_int> colidx_;      // for HYPRE
    std::vector<custom_int> rows_;        // for HYPRE
    std::vector<custom_int> nnz_v_;       // for HYPRE
    std::vector<int> row_ptr_;            // for AMGX
    std::vector<long long int> g_colidx_; // for AMGX

    int local_size_, global_size_, nnz_sum_, ilower_, iupper_, nnz_;
};
/*  10  8 0 0
 *   8 10 8 0
 *   0 8 10 8
 *   0 0 8 10
 */
// rows_= {0,1,2,3}, colidx_ = {0,1,0,1,2,1,2,3,2,3}, nnz_v_ ={2,3,3,2} using 1 rank
/*
A =
   10    8    0    0    0    0    0    0    0    0
    8   10    8    0    0    0    0    0    0    0
    0    8   10    8    0    0    0    0    0    0
    0    0    8   10    8    0    0    0    0    0
    0    0    0    8   10    8    0    0    0    0
    0    0    0    0    8   10    8    0    0    0
    0    0    0    0    0    8   10    8    0    0
    0    0    0    0    0    0    8   10    8    0
    0    0    0    0    0    0    0    8   10    8
    0    0    0    0    0    0    0    0    8   10
rows_={0,1,2},   colidx_={0,1,0,1,2,1,2,3},      nnz_v_={2,3,3},  row_ptr_={0,2,5,8}    at rank = 0
rows_={3,4,5},   colidx_={2,3,4,4,4,5,4,5,6},    nnz_v_={3,3,3},  row_ptr_={0,3,6,9}    at rank = 1
rows_={6,7,8,9}, colidx_={5,6,7,6,7,8,7,8,9,8,9},nnz_v_={3,3,3,2},row_ptr_={0,3,6,9,11} at rank = 2
*/

struct rhs {
    std::vector<double> values_;
    std::vector<custom_int> rows_;
};

class mtrx_reader{
public:
    mtrx_reader() = default;
    ~mtrx_reader() = default;
    void from_mtx(std::string const &fname, bool const &isSym, mtrx_csr &spdata,
            int const &myid, int const &num_procs);
    void set_b(mtrx_csr const &spdata, rhs &b_v, int const &myid, int const &num_procs);
private:
};

#endif /* SRC_MAIN_DATASET_H_ */
