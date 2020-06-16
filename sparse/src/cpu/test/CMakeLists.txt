include_directories(${GTEST_HOME}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${GTEST_HOME}/lib)
#
add_executable(test_sym_hypre     test_sym_matrix.cpp)
add_executable(test_unsym_hypre   test_unsym_matrix.cpp)
target_link_libraries(test_sym_hypre     reader_lib cpu_lib gtest pthread)
target_link_libraries(test_unsym_hypre   reader_lib cpu_lib gtest pthread)
#
add_test(NAME hypre_sym_rank1     COMMAND "mpirun" "-n" 1 "test_sym_hypre")
add_test(NAME hypre_sym_rank2     COMMAND "mpirun" "-n" 2 "test_sym_hypre")
add_test(NAME hypre_sym_rank3     COMMAND "mpirun" "-n" 3 "test_sym_hypre")
add_test(NAME hypre_unsym_rank1   COMMAND "mpirun" "-n" 1 "test_unsym_hypre")
add_test(NAME hypre_unsym_rank2   COMMAND "mpirun" "-n" 2 "test_unsym_hypre")
add_test(NAME hypre_unsym_rank3   COMMAND "mpirun" "-n" 3 "test_unsym_hypre")
set_tests_properties(hypre_sym_rank1 PROPERTIES LABELS "cpu_test;hypre_sym_test")
set_tests_properties(hypre_sym_rank2 PROPERTIES LABELS "cpu_test;hypre_sym_test")
set_tests_properties(hypre_sym_rank3 PROPERTIES LABELS "cpu_test;hypre_sym_test")
set_tests_properties(hypre_unsym_rank1 PROPERTIES LABELS "cpu_test;hypre_unsym_test")
set_tests_properties(hypre_unsym_rank2 PROPERTIES LABELS "cpu_test;hypre_unsym_test")
set_tests_properties(hypre_unsym_rank3 PROPERTIES LABELS "cpu_test;hypre_unsym_test")
#
set(input_data
    sym10.mtx
    unsym10.mtx
)
foreach(input1 ${input_data})
    file(COPY ${input1} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
