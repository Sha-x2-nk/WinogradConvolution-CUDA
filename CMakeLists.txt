cmake_minimum_required(VERSION 3.22)

project(WINOGRAD_CUDA LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(include)

set(CMAKE_CXX_FLAGS "${CMAKE_CUDA_FLAGS} /openmp:llvm")

set(SOURCES
    main.cu
)

add_executable(main ${SOURCES})
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main cublas cudnn OpenMP::OpenMP_CXX)