#ifndef SRC_MAIN_READER_H_
#define SRC_MAIN_READER_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

struct mtrx_csr {
    std::vector<double> values_;
    std::vector<int> rows_, colidx_;
    int local_size_, global_size_, nzz_sum_, ilower_, iupper_, nnz_;
};


class mtrx_reader{
public:
    mtrx_reader() = default;
    ~mtrx_reader() = default;
    void from_mtx(std::string const &fname, mtrx_csr &spdata,
            int const &myid, int const &num_procs);
private:

};


#endif /* SRC_MAIN_DATASET_H_ */
