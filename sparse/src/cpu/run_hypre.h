#include <vector>
#include <main/reader.h>
class HYPRE_solver{
public:
    HYPRE_solver() = default;
    ~HYPRE_solver() = default;
    void run_hypre(mtrx_csr const &spdata, const int &n, const std::vector<double> &Aex,
                    const std::vector<double> &bex);
    void return_x(std::vector<double> &x);
private:
    //std::vector<double> A_;
    //std::vector<double> b_;
};
