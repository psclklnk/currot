cmake_minimum_required(VERSION 3.4...3.18)
project(nadaraya_watson)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(pybind11)
pybind11_add_module(nadaraya_watson interface.cpp)
target_include_directories(nadaraya_watson PRIVATE include)
target_link_libraries(nadaraya_watson PRIVATE OpenMP::OpenMP_CXX)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(nadaraya_watson
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})