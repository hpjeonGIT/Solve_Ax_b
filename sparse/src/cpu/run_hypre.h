#include <vector>
#include "main/reader.h"

class HYPRE_solver{
public:
    HYPRE_solver() = default;
    ~HYPRE_solver() = default;
    void run_hypre(mtrx_csr &spdata, rhs &b_v, int const &myid);
    void get_result(std::vector<double> &x);
private:
    std::vector<double> x_values_;
};
