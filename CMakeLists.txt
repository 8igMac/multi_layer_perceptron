cmake_minimum_required(VERSION 3.0.0)
project(neuro_network)
add_executable(unit_test unit_test/data_set_test.cpp)
target_include_directories(unit_test include)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS  "-std=c++17")
