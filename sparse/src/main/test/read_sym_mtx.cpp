//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include "gtest/gtest.h"
#include "mpi.h"
#include "main/reader.h"


class ReadSymTest : public ::testing::Test {
protected:
    int myid, num_procs;
    ReadSymTest() {}
    virtual ~ReadSymTest() {}
    virtual void SetUp() {
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    }
    virtual void TearDown() {
        MPI_Finalize();
    }
};

TEST_F(ReadSymTest, testingSizeofData_rand) {
    mtrx_reader parsor;
    mtrx_csr spdata;
    parsor.from_mtx("sym10.mtx", true, spdata, myid, num_procs);
    int local_sum;
    MPI_Allreduce(&spdata.local_size_, &local_sum, 1, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD);
    EXPECT_EQ(local_sum, spdata.global_size_);
    EXPECT_EQ(28, spdata.nnz_sum_);
}

int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
