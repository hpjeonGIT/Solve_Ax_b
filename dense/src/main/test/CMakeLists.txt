include_directories(${GTEST_HOME}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${GTEST_HOME}/lib)
#
add_executable(dataset_Ab_size_test   dataset_Ab_size_test.cpp)
add_executable(dataset_Ab_return_test dataset_Ab_return_test.cpp)
target_link_libraries(dataset_Ab_size_test    data_lib gtest pthread)
target_link_libraries(dataset_Ab_return_test  data_lib gtest pthread)
#
add_test(NAME dataset_test_size   COMMAND dataset_Ab_size_test)
add_test(NAME dataset_test_return COMMAND dataset_Ab_return_test)
