//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "main/dataset.h"

class DataTest : public ::testing::Test {
protected:
    int N = 123;
    std::vector<double> A_, b_;
    Dataset s;
    DataTest() {}
    virtual ~DataTest() {}
    virtual void SetUp() {
        s.sym_matrix(N);
        A_.resize(N*N);
        b_.resize(N);
        A_ = s.return_A();
        b_ = s.return_b();
    }
    virtual void TearDown() {
        A_.clear();
        b_.clear();
    }
};

TEST_F(DataTest, testingReturnData_A) {
    EXPECT_EQ(N, A_[0]);
    EXPECT_EQ(N, A_.back());
    EXPECT_EQ(1, A_[N-1]);
}

TEST_F(DataTest, testingReturnData_b) {
    EXPECT_EQ(N,        b_[0]);
    EXPECT_EQ(abs(N-2), b_.back());
}

int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
