cmake_minimum_required(VERSION 3.10)
project(PCB_Anomaly_Detection_Production LANGUAGES CXX CUDA)

# Enforce Modern Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optimize compilation for target GPU architecture (RTX 3050 Ti - Ampere SM 86)
set(CMAKE_CUDA_ARCHITECTURES "86")

# Global Release Optimizations (Forces maximum compiler speedups, strips debug flags)
set(CMAKE_BUILD_TYPE RELEASE)

# ==========================================
# FIND DEPENDENCIES
# ==========================================
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs cudawarping cudaarithm cudafilters cudaimgproc)
find_package(CUDA REQUIRED)

# Dynamically locate TensorRT shared libraries and headers inside the Docker environment
find_path(NVINFER_INCLUDE_DIR NvInfer.h PATHS /usr/include/x86_64-linux-gnu /usr/local/cuda/include)
find_library(NVINFER_LIB nvinfer PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

# Dynamically locate open62541 from the system installation path (Handles .a and .so seamlessly)
find_path(OPEN62541_INCLUDE_DIR open62541.h PATHS /usr/local/include)
find_library(OPEN62541_LIB NAMES open62541 libopen62541 PATHS /usr/local/lib)

# ==========================================
# OBJECT LIBRARIES & HEADER COMPILATION
# ==========================================
# CRITICAL SOURCE INTERLOCK: Pulling kernel.cu relatively from the parent cpp folder
add_library(cuda_kernels OBJECT kernel.cu)
set_target_properties(cuda_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Configure global include directory spaces
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS} 
    ${NVINFER_INCLUDE_DIRS}
    ${OPEN62541_INCLUDE_DIR}
)

# ==========================================
# TARGET: PRODUCTION ENGINE EXECUTABLE
# ==========================================
# Compile your production code directly into the final "pcb_engine" binary
add_executable(pcb_engine main.cpp)

# Map dynamic include requirements (including host CUDA Toolkit directories)
target_include_directories(pcb_engine PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${NVINFER_INCLUDE_DIR}
)

# Link explicitly to required runtime object libraries and dependency layers
target_link_libraries(pcb_engine PRIVATE 
    cuda_kernels 
    cudart 
    ${NVINFER_LIB} 
    ${OpenCV_LIBS}
    ${OPEN62541_LIB}
)
