cmake_minimum_required(VERSION 3.14)
project(classifim_libs)

# Enable C++23 or later
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(xtensor REQUIRED)

# Prefix is removed since we are already in that directory:
# classifim/cpp_src/fil24_hamiltonian.cpp
set(SOURCE_FILES
  metric.cpp
  metric.h
)

set(WRAPPER_FILES
  metric_wrapper.h
  metric_wrapper.cpp
)

# Add a shared library target for classifim.so
add_library(classifim SHARED ${SOURCE_FILES} ${WRAPPER_FILES})
target_include_directories(classifim PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(classifim PUBLIC xtensor)
set_target_properties(classifim PROPERTIES OUTPUT_NAME "classifim")
# Or is it classifim.dll?
if(WIN32)
  set_target_properties(classifim PROPERTIES SUFFIX ".dll")
else()
  # Enable warnings:
  target_compile_options(classifim PRIVATE -Wall -Wextra -pedantic)
endif()
