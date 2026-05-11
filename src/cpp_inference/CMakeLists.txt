cmake_minimum_required(VERSION 3.10)
project(PCB_Anomaly_Detection)

set(CMAKE_CXX_STANDARD 17)

# Find Dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs cudawarping cudaarithm cudafilters cudaimgproc)
find_package(CUDA REQUIRED)

# Manually find TensorRT (it usually doesn't have a find_package script)
find_library(NVINFER libnvinfer.so PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(pre preprocess.cpp)
target_link_libraries(pre ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${NVINFER})
