cmake_minimum_required(VERSION 3.22)
project(CPSO_CPlus)

set(CMAKE_CXX_STANDARD 14)


find_path(EIGEN_DIR NAMES signature_of_eigen3_matrix_library
        PATHS
        ./components/eigen
        PATH_SUFFIXES eigen3 eigen
        )
include_directories(${EIGEN_DIR})

include_directories(.)

add_executable(CPSO_CPlus
        main.cpp "components/cpso optimizations/optimizer.cpp" "components/cpso optimizations/optimizer.h" "components/data loader/data_provider.cpp" "components/data loader/data_provider.h" components/data_types.h)