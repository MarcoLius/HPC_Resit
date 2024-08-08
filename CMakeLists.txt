cmake_minimum_required(VERSION 3.26)
project(Resit C)

set(CMAKE_C_STANDARD 17)

find_package(MPI REQUIRED)
include_directories("/opt/homebrew/Cellar/mpich/4.1.2/bin")
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)
add_executable(Resit part2.c)