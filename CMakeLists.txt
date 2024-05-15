cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "-O3")

find_package(Torch REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75 -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr")
# --use-fast-math

add_executable(example-app main.cpp grad_hash.cu)

set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
set_property(TARGET example-app PROPERTY CUDA_STANDARD 17)

target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_include_directories(example-app PUBLIC ${TORCH_INCLUDE_DIRS})