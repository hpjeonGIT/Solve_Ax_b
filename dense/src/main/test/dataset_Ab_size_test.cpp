//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include "gtest/gtest.h"
#include "main/dataset.h"

class DataTest : public ::testing::Test {
protected:
    int N = 123;
    Dataset r, s;
    DataTest() {}
    virtual ~DataTest() {}
    virtual void SetUp() {
    	r.random_matrix(N);
    	s.sym_matrix(N);
    }
    virtual void TearDown() {}
};

TEST_F(DataTest, testingSizeofData_rand) {
	EXPECT_EQ(N*N, r.return_size_A());
	EXPECT_EQ(N,   r.return_size_b());
}

TEST_F(DataTest, testingSizeofData_sym) {
	EXPECT_EQ(N*N, s.return_size_A());
	EXPECT_EQ(N,   s.return_size_b());
}

int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
