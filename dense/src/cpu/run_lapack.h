#include <vector>
class CPU_solver{
public:
    CPU_solver() = default;
    ~CPU_solver() = default;
    void run_lapack(const int &n, const int &m);
    void deliver_result(std::vector<double> &x);
private:
    std::vector<double> A_;
    std::vector<double> b_;    
};
