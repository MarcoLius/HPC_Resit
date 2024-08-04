cmake_minimum_required(VERSION 3.26)
project(Resit C)

set(CMAKE_C_STANDARD 17)

set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/17.0.6/include")	#你的libomp/include	如果是按步骤来的，那不需要改
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/opt/homebrew/Cellar/libomp/17.0.6/lib/libomp.dylib")	#你的libomp.dylib 如果是按步骤来的，那不需要改

set(CMAKE_OSX_ARCHITECTURES "x86_64")

find_package(OpenMP REQUIRED)

if(OpenMP_C_FOUND)
    #增加编译器的头文件搜索路经
    include_directories(${OpenMP_C_INCLUDE_DIRS})
    #添加可执行文件
    add_executable(Resit part1.c)
    #链接动态库文件到目标可执行文件中
    target_link_libraries(Resit PUBLIC OpenMP::OpenMP_C)
endif()
