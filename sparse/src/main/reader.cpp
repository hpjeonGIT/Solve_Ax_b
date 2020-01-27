#include "reader.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

void mtrx_reader::from_mtx(std::string const &fname, bool const &isSym, mtrx_csr &spdata,
        int const &myid, int const &num_procs){
    std::ifstream file_obj;
    file_obj.open(fname, std::ios::in);
    bool afterheader=false;
    int nrow, ncol;
    double val;

    struct rcv{
    int row;
    int col;
    double val;
    };
    std::vector<rcv> rowsort; rcv p;
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
                p.row = nrow; p.col = ncol; p.val = val;
                rowsort.push_back(p);
                if (nrow != ncol && isSym) {
                    p.row = ncol; p.col = nrow; p.val = val;
                    rowsort.push_back(p);
                }

            }
        }
        file_obj.close();
    }
// sorting for CSR format
    std::sort(rowsort.begin(), rowsort.end(),
          [](const auto& i, const auto& j) { return i.row < j.row; } );
    struct cv{
       int col;
       double val;
       };
   std::vector<cv> colsort;
   int istart, iend, nrow0, isize;
   nrow0 = rowsort[0].row;
   istart = 0;
   p.row = nrow+999; p.col = ncol+999; p.val = -9999.;
   rowsort.push_back(p); //make the last data as dummy to finish the loop just before the end
   for (int i=1; i<rowsort.size(); i++) {
        nrow = rowsort[i].row;
        if (nrow != nrow0 ) {
            isize = i - istart;
            colsort.resize(isize);
            // copy col data of same row
            for (int n=istart; n< i; n++) {
                colsort[n-istart].col = rowsort[n].col;
                colsort[n-istart].val = rowsort[n].val;
            }
            // sorting col data
            std::sort(colsort.begin(), colsort.end(),
                      [](const auto& i, const auto& j) { return i.col < j.col; } );
            // copy back the sorted data
            for (int n=istart; n< i; n++) {
                rowsort[n].col = colsort[n-istart].col;
                rowsort[n].val = colsort[n-istart].val;
            }
            nrow0 = nrow;
            istart = i;
        }
   }
   // Now rowsort has contiguous/ascending order of row/colindx data
   // Slice CSR data onto each domain of MPI rank
   for (int i=0; i<rowsort.size()-1; i++) {
       nrow = rowsort[i].row; ncol = rowsort[i].col; val = rowsort[i].val;
       if (nrow >= spdata.ilower_ && nrow <= spdata.iupper_) {
           spdata.rows_.push_back(nrow); spdata.colidx_.push_back(ncol);
           spdata.values_.push_back(val); spdata.nnz_++;
        }
    }
   for (int i=0; i< 10; i++) std::cout << spdata.rows_[i];
   std::cout << std::endl;
   for (int i=0; i< 10; i++) std::cout << spdata.colidx_[i];
   std::cout << std::endl;
   for (int i=0; i< 10; i++) std::cout << spdata.values_[i];
   std::cout << std::endl;

}
