//Ref: https://stackoverflow.com/questions/26030700/unit-testing-c-setup-and-teardown
#include "gtest/gtest.h"
#include "mpi.h"
#include "main/reader.h"
#include "gpu/run_amgx.h"


class TestUnSymSolver : public ::testing::Test {
protected:
    int myid, num_procs;
    TestUnSymSolver() {}
    virtual ~TestUnSymSolver() {}
    virtual void SetUp() {
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    }
    virtual void TearDown() {
        MPI_Finalize();
    }
};

TEST_F(TestUnSymSolver, testingUnSymAMGX) {
    mtrx_reader parsor;
    mtrx_csr spdata;
    rhs  b_v;
    parsor.from_mtx("unsym10.mtx", false, spdata, myid, num_procs);
    AMGX_solver gpusolver;
    parsor.set_b(spdata, b_v, myid, num_procs);
    gpusolver.run_amgx(spdata, b_v, myid, num_procs);
    std::vector<double> x(spdata.local_size_,0);
    gpusolver.get_result(x);
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
    double tol = 1.e-5;
    EXPECT_NEAR(x_all[0],-3.25058, tol);
    EXPECT_NEAR(x_all[1], 4.18823, tol);
    EXPECT_NEAR(x_all[2],-2.14103, tol);
    EXPECT_NEAR(x_all[3],-0.61342, tol);
    EXPECT_NEAR(x_all[4], 3.14017, tol);
    EXPECT_NEAR(x_all[5],-2.76347, tol);
    EXPECT_NEAR(x_all[6], 1.45669, tol);
    EXPECT_NEAR(x_all[7], 1.47217, tol);
    EXPECT_NEAR(x_all[8],-2.11482, tol);
    EXPECT_NEAR(x_all[9], 2.48038, tol);
}

int main(int argc, char** argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
