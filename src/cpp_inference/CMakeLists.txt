cmake_minimum_required(VERSION 3.10)
project(PCB_Anomaly_Detection LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Set target GPU architecture for RTX 3050 Ti (Ampere SM 86)
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_STANDARD 17)

# Find Dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs cudawarping cudaarithm cudafilters cudaimgproc)
find_package(CUDA REQUIRED)

# Manually find TensorRT 
#find_library(NVINFER libnvinfer.so PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_path(NVINFER_INCLUDE_DIR NvInfer.h PATHS /usr/include/x86_64-linux-gnu /usr/local/cuda/include)
find_library(NVINFER_LIB nvinfer PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

# Compile the CUDA kernel independently as an object library
add_library(cuda_kernels OBJECT kernel.cu)
set_target_properties(cuda_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${NVINFER_INCLUDE_DIRS})

# Dynamic debug executable target
add_executable(tensorrt tensorrt.cpp)
# EXPLICIT SOLUTION: Map the CUDA Toolkit include directory to the host CXX compiler target
target_include_directories(tensorrt PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${NVINFER_INCLUDE_DIR})
target_link_libraries(tensorrt cuda_kernels ${OpenCV_LIBS} ${NVINFER_LIB} cuda cudart)

add_executable(pre preprocess.cpp)
target_link_libraries(pre ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${NVINFER})
