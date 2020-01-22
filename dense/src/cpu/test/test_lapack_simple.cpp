//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "cpu/run_lapack.h"

class LapackTest : public ::testing::Test {
protected:
    CPU_solver csolver;
    std::vector<double> A_, b_, x_;
    LapackTest() {}
    virtual ~LapackTest() {}
    virtual void SetUp() {
    }
    virtual void TearDown() {
        A_.clear();
        b_.clear();
        x_.clear();
    }
};

TEST_F(LapackTest, testingSimpleMatrix1) {
    A_ = {1., 0., 0., 1.};
    b_ = {1., 2.};
    csolver.run_lapack(2, A_, b_);
    x_.resize(2);
    csolver.deliver_result(x_);
    EXPECT_EQ(x_[0], 1.);
    EXPECT_EQ(x_[1], 2.);
}

TEST_F(LapackTest, testingSimpleMatrix2) {
    A_ = {1., 0., 0., 2.};
    b_ = {1., 2.};
    csolver.run_lapack(2, A_, b_);
    x_.resize(2);
    csolver.deliver_result(x_);
    EXPECT_EQ(x_[0], 1.);
    EXPECT_EQ(x_[1], 1.);
}

int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
