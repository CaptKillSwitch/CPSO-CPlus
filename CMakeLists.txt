cmake_minimum_required(VERSION 3.22)
project(CPSO_CPlus)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(CPSO_CPlus
        main.cpp "components/cpso optimizations/optimizer.cpp" "components/cpso optimizations/optimizer.h" "components/data loader/data_loader.cpp" "components/data loader/data_loader.h")
