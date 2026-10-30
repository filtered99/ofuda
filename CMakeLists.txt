cmake_minimum_required(VERSION 3.12)
project(OfudaEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional CUDA Support Integration
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "[Ofuda] CUDA Toolkit found: " ${CUDAToolkit_VERSION})
    add_definitions(-DUSE_CUDA)
else()
    message(STATUS "[Ofuda] CUDA Toolkit not found. Compiling for CPU fallback.")
endif()

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

add_executable(ofuda ofuda.cpp)
target_link_libraries(ofuda PRIVATE dlib::dlib)
