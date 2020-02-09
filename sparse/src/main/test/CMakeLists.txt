include_directories(${GTEST_HOME}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${GTEST_HOME}/lib)
#
add_executable(test_read_sym_mtx   read_sym_mtx.cpp)
add_executable(test_read_unsym_mtx read_unsym_mtx.cpp)
target_link_libraries(test_read_sym_mtx   reader_lib gtest pthread)
target_link_libraries(test_read_unsym_mtx  reader_lib gtest pthread)
#
add_test(NAME reader_sym_rank1   COMMAND "mpirun" "-n" 1 "test_read_sym_mtx")
add_test(NAME reader_sym_rank2   COMMAND "mpirun" "-n" 2 "test_read_sym_mtx")
add_test(NAME reader_sym_rank3   COMMAND "mpirun" "-n" 3 "test_read_sym_mtx")
add_test(NAME reader_unsym_rank1 COMMAND "mpirun" "-n" 1 "test_read_unsym_mtx")
add_test(NAME reader_unsym_rank2 COMMAND "mpirun" "-n" 2 "test_read_unsym_mtx")
add_test(NAME reader_unsym_rank3 COMMAND "mpirun" "-n" 3 "test_read_unsym_mtx")
#
set(input_data
    sym10.mtx
    unsym10.mtx
)
foreach(input1 ${input_data})
    file(COPY ${input1} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
