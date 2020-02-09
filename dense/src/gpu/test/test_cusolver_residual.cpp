//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "gpu/run_cusolver.h"

class CuSolverTest : public ::testing::Test {
protected:
    GPU_solver gsolver;
    std::vector<double> A_, b_, x_, b0_;
    CuSolverTest() {}
    virtual ~CuSolverTest() {}
    virtual void SetUp() {
    }
    virtual void TearDown() {
        A_.clear();
        b_.clear();
        x_.clear();
        b0_.clear();
    }
};

TEST_F(CuSolverTest, testingResidual1) {
    A_ = {1., 0., 0., 1.};
    b_ = {1., 2.};
    b0_ = b_;
    gsolver.run_cuda_symsolver(2, A_, b_);
    x_.resize(2);
    gsolver.deliver_result(x_);
    double lsum;
    double resid = 0.0;
    for (int i=0;i<2; i++) {
        lsum = 0.0;
        for (int j=0;j<2; j++){
            lsum += A_[i*2 + j] *x_[i];
        }
        resid += lsum - b0_[i];
    }
    double tol=1.e-6;
    EXPECT_NEAR(resid, 0.0, tol);
}


int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
