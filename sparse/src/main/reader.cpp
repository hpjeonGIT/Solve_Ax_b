#include "reader.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

void mtrx_reader::from_mtx(std::string const &fname, mtrx_csr &spdata,
        int const &myid, int const &num_procs){
    std::ifstream file_obj;
    file_obj.open(fname, std::ios::in);
    bool afterheader=false;
    int nrow, ncol;
    double val;
    if (file_obj.is_open()) {
        std::string line;
        while(std::getline(file_obj, line)) {
            //std::cout << line.c_str() << std::endl;
            if (line[0] != '%' && !afterheader) {
                afterheader = true;
                std::stringstream stream(line);
                stream >> spdata.global_size_; stream >> spdata.global_size_;
                stream >> spdata.nzz_sum_;
                spdata.local_size_ = static_cast<int>(round(spdata.global_size_/num_procs));
                spdata.ilower_ = myid*spdata.local_size_;
                spdata.iupper_ = spdata.ilower_ + spdata.local_size_ - 1; // 0th index
                if (myid == (num_procs-1)) {
                    spdata.iupper_ = spdata.global_size_ - 1;
                }
                spdata.local_size_ = spdata.iupper_ - spdata.ilower_ + 1;
                std::cout << "myid=" << myid << " data range " << spdata.ilower_ << " "
                        << spdata.iupper_ << std::endl;
                spdata.nnz_ = 0;
            }
            if (afterheader) {
                std::stringstream stream(line);
                stream >> nrow; stream >> ncol ; stream >> val;
                nrow -= 1; ncol -= 1; // 0th index
                if (nrow >= spdata.ilower_ && nrow <= spdata.iupper_) {
                    spdata.rows_.push_back(nrow); spdata.colidx_.push_back(ncol);
                    spdata.values_.push_back(val); spdata.nnz_++;

                }
            }
        }
        file_obj.close();
    }
}
