//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include "gtest/gtest.h"
#include "mpi.h"
#include "main/reader.h"
#include "cpu/run_hypre.h"


class TestSymSolver : public ::testing::Test {
protected:
    int myid, num_procs;
    TestSymSolver() {}
    virtual ~TestSymSolver() {}
    virtual void SetUp() {
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    }
    virtual void TearDown() {
        MPI_Finalize();
    }
};

TEST_F(TestSymSolver, testingSymHYPRE) {
    mtrx_reader parsor;
    mtrx_csr spdata;
    rhs  b_v;
    parsor.from_mtx("sym10.mtx", true, spdata, myid, num_procs);
    HYPRE_solver cpusolver;
    parsor.set_b(spdata, b_v, myid, num_procs);
    cpusolver.run_hypre(spdata, b_v, myid);
    std::vector<double> x(spdata.local_size_,0);
    cpusolver.get_result(x);
    std::vector<int> ncount(num_procs,0), ndisp(num_procs,0);
    std::vector<double> x_all(spdata.global_size_,0);
    MPI_Allgather(&spdata.local_size_,1, MPI_INT,
            ncount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i=1;i< num_procs; i++){
        ndisp[i] = ndisp[i-1] + ncount[i-1];
    }
    std::cout << "ndisp" << std::endl;
    MPI_Allgatherv(x.data(), spdata.local_size_, MPI_DOUBLE,
            x_all.data(), ncount.data(), ndisp.data(), MPI_DOUBLE,
            MPI_COMM_WORLD);
    double tol = 1.e-6;
    EXPECT_NEAR(x_all[0],  0.834992, tol);
    EXPECT_NEAR(x_all[1], -0.918740, tol);
    EXPECT_NEAR(x_all[2],  0.563433, tol);
    EXPECT_NEAR(x_all[3],  0.589449, tol);
    EXPECT_NEAR(x_all[4], -0.800244, tol);
    EXPECT_NEAR(x_all[5],  1.035856, tol);
    EXPECT_NEAR(x_all[6],  0.255424, tol);
    EXPECT_NEAR(x_all[7], -0.480136, tol);
    EXPECT_NEAR(x_all[8],  1.344746, tol);
    EXPECT_NEAR(x_all[9], -0.075797, tol);
}

int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
