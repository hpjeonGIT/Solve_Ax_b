/*
 * dataset.h
 *
 *  Created on: Jan 19, 2020
 *      Author: hpjeon
 */

#ifndef SRC_MAIN_DATASET_H_
#define SRC_MAIN_DATASET_H_

#include <vector>
class Dataset{
public:
    Dataset() = default;
    ~Dataset() = default;
    void random_matrix(const int &n);
    void sym_matrix   (const int &n);
    std::vector<double> return_A();
    std::vector<double> return_b();
private:
    std::vector<double> A_;
    std::vector<double> b_;
};


#endif /* SRC_MAIN_DATASET_H_ */
