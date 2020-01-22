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
    int return_size_A();
    int return_size_b();
private:
    std::vector<double> A_;
    std::vector<double> b_;
};

int Dataset::return_size_A(){
    return A_.size();
}

int Dataset::return_size_b(){
    return b_.size();
}

#endif /* SRC_MAIN_DATASET_H_ */
