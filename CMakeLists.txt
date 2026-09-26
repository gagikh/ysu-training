# Build for the whole course.
#
#   cmake -B build
#   cmake --build build -j
#   ./build/day01
#
# 3.24 is the first CMake that accepts CUDA_ARCHITECTURES native.
cmake_minimum_required(VERSION 3.24)
project(cuda-course-ysu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# native asks the driver which card is in this machine and compiles SASS for
# that architecture only. Day 1 covers what that does and does not give you:
# the binary will not run on a different generation, and carries no PTX to
# JIT from. Override it for a machine whose GPU is not the build machine's:
#
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
#
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# -lineinfo lets Nsight Compute map SASS back to source lines. Needed from
# Day 4 on, and it does not slow the code down.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui)

message(STATUS "CUDA        ${CUDAToolkit_VERSION}")
message(STATUS "OpenCV      ${OpenCV_VERSION}")
message(STATUS "GPU arch    ${CMAKE_CUDA_ARCHITECTURES}")

# Days 1-4 need only the CUDA runtime.
foreach(day day01 day02 day03 day04)
  add_executable(${day} ${day}/template.cu)
  target_include_directories(${day} PRIVATE ${CMAKE_SOURCE_DIR})
endforeach()

# Days 5-8 read an image from disk and write the result back.
foreach(day day05 day06 day07 day08)
  add_executable(${day} ${day}/template.cu)
  target_include_directories(${day} PRIVATE ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${day} PRIVATE ${OpenCV_LIBS})
endforeach()

# Day 9 additionally calls cuBLAS and NPP.
add_executable(day09 day09/template.cu)
target_include_directories(day09 PRIVATE ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(day09 PRIVATE ${OpenCV_LIBS} CUDA::cublas CUDA::nppc CUDA::nppif)
