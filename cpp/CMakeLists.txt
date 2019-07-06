cmake_minimum_required(VERSION 2.6)
project(nn_from_scratch)

set(CMAKE_CXX_STANDARD 11)
add_executable(main main.cpp)

if(MSVC)
    target_compile_options(main PUBLIC /arch:AVX2)
else()
    target_compile_options(main PUBLIC -march=native)
endif()
