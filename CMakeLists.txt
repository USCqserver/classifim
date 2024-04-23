# Scikit-build requires CMake 3.15
cmake_minimum_required(VERSION 3.15)
project(classifim)

# Example usages:
# 1. Run classifim tests (requires Catch2v3):
#   mkdir build && cd build && cmake -DBUILD_TESTS=ON .. && cmake --build . \
#     && tests/cpp_src/tests
# 2. Build classifim shared library:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Debug .. && cmake --build .
# 3. Release build classifim shared library:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release .. \
#     && cmake --build .
# 4. Cross-platform release build classifim shared library on Linux targeting Windows:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/x86_64-w64-mingw32.cmake .. \
#     && cmake --build .
# 5. You may want to copy that library after compilation:
#   # assuming you are in build directory and ran #2 or #3:
#   cp src/classifim/cpp_src/libclassifim.so ../src/classifim/lib/
#   # for #4:
#   cp src/classifim/cpp_src/libclassifim.dll* ../src/classifim/lib/
# 6. Cleanup
#   cd .. # if you are in build directory
#   rm -r build

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(BUILD_SHARED_LIBS)
  add_subdirectory(src/classifim/cpp_src)
  # If ${SKBUILD_PLATLIB_DIR} is defined, we need to install the shared library
  # (compiled in src/classifim/cpp_src) to that directory:
  if(DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS classifim
      LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/classifim/lib
      RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/classifim/lib
    )
  endif()
endif()

if(BUILD_TESTS)
  add_subdirectory(tests/cpp_src)
endif()
