//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "gpu/run_cusolver.h"

class CuSolverTest : public ::testing::Test {
protected:
    GPU_solver gsolver;
    std::vector<double> A_, b_, x_;
    CuSolverTest() {}
    virtual ~CuSolverTest() {}
    virtual void SetUp() {
    }
    virtual void TearDown() {
        A_.clear();
        b_.clear();
        x_.clear();
    }
};

TEST_F(CuSolverTest, testingSimpleMatrix1) {
    A_ = {1., 0., 0., 1.};
    b_ = {1., 2.};
    gsolver.run_cuda_symsolver(2, A_, b_);
    x_.resize(2);
    gsolver.deliver_result(x_);
    double tol = 1.e-6;
    EXPECT_NEAR(x_[0], 1., tol);
    EXPECT_NEAR(x_[1], 2., tol);
}

TEST_F(CuSolverTest, testingSimpleMatrix2) {
    A_ = {1., 0., 0., 2.};
    b_ = {1., 2.};
    gsolver.run_cuda_symsolver(2, A_, b_);
    x_.resize(2);
    gsolver.deliver_result(x_);
    double tol = 1.e-6;
    EXPECT_NEAR(x_[0], 1., tol);
    EXPECT_NEAR(x_[1], 1., tol);
}

int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
