#include <vector>
#include "main/reader.h"

class AMGX_solver{
public:
    AMGX_solver() = default;
    ~AMGX_solver() = default;
    void run_amgx(mtrx_csr &spdata, rhs &b_v, int const &myid,
            int const &num_procs);
    void get_result(std::vector<double> &x);
private:
    std::vector<double> x_values_;
};
