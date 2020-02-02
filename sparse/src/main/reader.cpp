#include "reader.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "mpi.h"

void mtrx_reader::from_mtx(std::string const &fname, bool const &isSym, mtrx_csr &spdata,
        int const &myid, int const &num_procs){
    std::ifstream file_obj;
    file_obj.open(fname, std::ios::in);
    bool afterheader=false;
    int nrow, ncol, nnz_local;
    double val;

    struct rcv{
    int row;
    int col;
    double val;
    };
    std::vector<rcv> rowsort; rcv p; spdata.nnz_sum_ = 0;
    if (file_obj.is_open()) {
        std::string line;
        while(std::getline(file_obj, line)) {
            //std::cout << line.c_str() << std::endl;
            if (afterheader) {
                std::stringstream stream(line);
                if (!stream.str().empty()) { // check any empty line in the file
                    stream >> nrow; stream >> ncol ; stream >> val;
                    nrow -= 1; ncol -= 1; // 0th index
                    p.row = nrow; p.col = ncol; p.val = val;
                    rowsort.push_back(p); spdata.nnz_sum_ ++;
                    if (nrow != ncol && isSym) {
                        p.row = ncol; p.col = nrow; p.val = val;
                        rowsort.push_back(p); spdata.nnz_sum_ ++;
                    }
                }
            }
            if (line[0] != '%' && !afterheader) {
                afterheader = true;
                std::stringstream stream(line);
                stream >> spdata.global_size_; stream >> spdata.global_size_;
                int ntmp;
                stream >> ntmp; // when MM is symmetric, this is not real nnz_sum. Symm part only.
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
                spdata.nnz_sum_ = 0;
            }
        }
        file_obj.close();
    }

    /*
    for (int i=0; i<rowsort.size();i++) {
        std::cout << "rowsor" << rowsort[i].row << " " << rowsort[i].col << " " << rowsort[i].val << std::endl;
    }
    */
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
   nrow0 = spdata.ilower_ - 1; int nrow1 = nrow0 + 1;
   nnz_local = 0;
   for (int i=0; i<rowsort.size()-1; i++) {
       nrow = rowsort[i].row; ncol = rowsort[i].col; val = rowsort[i].val;
       if (nrow >= spdata.ilower_ && nrow <= spdata.iupper_) {
           if (nrow != nrow0) {
               spdata.rows_.push_back(nrow);
               nrow0 = nrow;
           }
           if (nrow != nrow1) {
               spdata.nnz_v_.push_back(nnz_local);
               nrow1 = nrow;
               nnz_local = 1;
           } else{
               nnz_local++;
           }
           //spdata.rows_.push_back(nrow);
           spdata.colidx_.push_back(ncol);
           spdata.values_.push_back(val);
           spdata.nnz_++;
        }
   }
   spdata.nnz_v_.push_back(nnz_local); // update of the last element

   for (int i=0; i<spdata.nnz_v_.size();i++) {
       std::cout << myid << "nnz_v" << spdata.nnz_v_[i] << std::endl;
   }

   //std::cout << "myid=" << myid << " " << nnz_ << " " << values_.size() << std::endl;
   if (spdata.values_.size() != spdata.nnz_) {
       std::cout << "something is wrong. nnz sum doesn't match \n";
       std::cout << "number of values =" << spdata.values_.size() << std::endl;
       std::cout << "nnz_ =" << spdata.nnz_ << std::endl;
       throw;
   }
   if (std::accumulate(spdata.nnz_v_.begin(), spdata.nnz_v_.end(),0)
       != spdata.nnz_) {
       std::cout << "something is wrong. nnz sum doesn't match2 \n";
       throw;
   }
   int nnz_sum;
   MPI_Allreduce(&spdata.nnz_, &nnz_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (nnz_sum != spdata.nnz_sum_ ) {
       std::cout << "something is wrong. sum of nnz_ doesn't match \n";
       std::cout << "number of nnz_sum =" << nnz_sum << std::endl;
       std::cout << "nnz_sum from parsing" << spdata.nnz_sum_ << std::endl;
       throw;
   }
}

void mtrx_reader::set_b(mtrx_csr const &spdata, rhs &b_v,
        int const &myid, int const &num_procs){
    b_v.rows_.resize(spdata.local_size_);
    b_v.values_.resize(spdata.local_size_,0.0);
    for (int i = 0; i < spdata.local_size_; i++) {
        b_v.rows_[i] = i + spdata.ilower_;
        b_v.values_[i] = static_cast<double> (b_v.rows_[i] + 1);
    }
}
