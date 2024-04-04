cmake_minimum_required(VERSION 3.14)
project(classifim_bench_libs)

# Enable C++23 or later
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(xtensor REQUIRED)

# Prefix is removed since we are already in that directory:
# classifim_bench/cpp_src/fil24_hamiltonian.cpp
set(SOURCE_FILES
  metric.cpp
  metric.h
)

set(WRAPPER_FILES
  metric_wrapper.h
  metric_wrapper.cpp
)

# Add a shared library target for classifim_bench.so
add_library(classifim_bench SHARED ${SOURCE_FILES} ${WRAPPER_FILES})
target_include_directories(classifim_bench PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(classifim_bench PUBLIC xtensor)
set_target_properties(classifim_bench PROPERTIES OUTPUT_NAME "classifim_bench")
# Or is it classifim_bench.dll?
if(WIN32)
  set_target_properties(classifim_bench PROPERTIES SUFFIX ".dll")
else()
  # Enable warnings:
  target_compile_options(classifim_bench PRIVATE -Wall -Wextra -pedantic)
endif()
